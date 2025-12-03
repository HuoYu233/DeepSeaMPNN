#!/usr/bin/env python3
"""
ProteinMPNN-thermo-finetuning
"""

import os
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import argparse
import wandb
from datetime import datetime
from training.model_utils import ProteinMPNN, featurize, loss_nll, NoamOpt


class ThermophileDataset(Dataset):
    """Thermo-protein dataset"""
    
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        print(f"thermo-protein dataset size: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_batch(batch):
    """Batch function"""
    return batch


def calculate_accuracy(S, log_probs, mask):
    """
    Calculate accuracy of the model predictions.
    Args:
        S: ground truth sequence [B, L]
        log_probs: predicted log probability [B, L, 21]
        mask: mask for valid position [B, L]
    """
    predicted = torch.argmax(log_probs, dim=-1)  # [B, L]
    correct = (predicted == S).float()  # [B, L]
    
    valid_positions = mask.sum() # mask = 1 valid
    if valid_positions > 0:
        accuracy = (correct * mask).sum() / valid_positions
    else:
        accuracy = torch.tensor(0.0, device=S.device)
    
    return accuracy.item()


def calculate_perplexity(log_probs, S, mask):
    """Calculate perplexity of the model predictions."""
    # get log_probs at true labels
    gathered_log_probs = torch.gather(log_probs, 2, S.unsqueeze(-1)).squeeze(-1)  # [B, L]
    
    valid_positions = mask.sum()
    if valid_positions > 0:
        avg_log_prob = (gathered_log_probs * mask).sum() / valid_positions
        perplexity = torch.exp(-avg_log_prob)
    else:
        perplexity = torch.tensor(float('inf'))
    
    return perplexity.item()


class ProteinMPNNTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Running on Device: {self.device}")
        
        self.model = ProteinMPNN(
            num_letters=21,
            node_features=args.hidden_dim,
            edge_features=args.hidden_dim,
            hidden_dim=args.hidden_dim,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            augment_eps=args.augment_eps,
            k_neighbors=args.k_neighbors,
            dropout=args.dropout
        ).to(self.device)

        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_ppl': [],
            'val_ppl': [],
            'lr': []
        }
        # load pretrained model
        if args.pretrained_model:
            print(f"Loading pretrained model: {args.pretrained_model}")
            checkpoint = torch.load(args.pretrained_model, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # set up finetuning strategy
        self._setup_finetuning_strategy()
        
        # set optimizer and scheduler
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
        print(f"Total params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Percentages: {100 * sum(p.numel() for p in trainable_params) / sum(p.numel() for p in self.model.parameters()):.2f}%")
        
        if args.optimizer == 'noam':
            self.optimizer = NoamOpt(
                args.hidden_dim, 2, args.warmup_steps,
                torch.optim.Adam(trainable_params, lr=0, betas=(0.9, 0.98), eps=1e-9),
                step=0
            )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params, 
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.999)
            )
        
        if args.optimizer != 'noam':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=args.scheduler_restart, T_mult=1
            )
        
        # init best metrics
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def _setup_finetuning_strategy(self):
        """set up finetuning strategy"""
        strategy = self.args.finetune_strategy
        
        # print("\n=== Model structure information ===")
        # for name, module in self.model.named_children():
        #     print(f"Module: {name}, type: {type(module).__name__}")
        # print("=====================\n")
        
        if strategy == 'full':
            print("Full params ft")
            for param in self.model.parameters():
                param.requires_grad = True
                
        elif strategy == 'decoder_only':
            print("Decoder-Only ft")

            for param in self.model.parameters():
                param.requires_grad = False
            
            if hasattr(self.model, 'decoder_layers'):
                for layer in self.model.decoder_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
                        
            if hasattr(self.model, 'W_out'):
                for param in self.model.W_out.parameters():
                    param.requires_grad = True
            
            if hasattr(self.model, 'W_s'):
                for param in self.model.W_s.parameters():
                    param.requires_grad = True
                    
        elif strategy == 'top_layers':
            print("Top layer encoder and full decoder ft")
            for param in self.model.parameters():
                param.requires_grad = False
            
            if hasattr(self.model, 'encoder_layers'):
                num_layers_to_finetune = min(2, len(self.model.encoder_layers))
                for layer in self.model.encoder_layers[-num_layers_to_finetune:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            
            if hasattr(self.model, 'decoder_layers'):
                for layer in self.model.decoder_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
            
            if hasattr(self.model, 'W_out'):
                for param in self.model.W_out.parameters():
                    param.requires_grad = True
            if hasattr(self.model, 'W_s'):
                for param in self.model.W_s.parameters():
                    param.requires_grad = True
                    
        elif strategy == 'feature_extractor_frozen':
            print("Transformers ft")
            for param in self.model.parameters():
                param.requires_grad = True
            
            modules_to_freeze = ['features', 'W_e']
            for name, module in self.model.named_children():
                if name in modules_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False
        
        elif strategy == 'progressive':
            print("Output layer ft")
            for param in self.model.parameters():
                param.requires_grad = False
            
            output_modules = ['W_out', 'W_s']
            for name, module in self.model.named_children():
                if name in output_modules:
                    for param in module.parameters():
                        param.requires_grad = True
        
        print("\n=== Trainable Params Information ===")
        trainable_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"{name}, Shape {param.shape}, Num of Params: {param.numel():,}")
        
        # print(f"\nTotal Params: {total_params:,}")
        # print(f"Trainable Params: {trainable_params:,}")
        # print(f"Trainable Percentages: {100 * trainable_params / total_params:.2f}%")
        # print("=======================\n")

    def train_epoch(self, dataloader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        total_perplexity = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
                    batch, self.device
                )
                
                # if batch_idx == 0:
                #     trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                #     if not trainable_params:
                #         raise ValueError("没有可训练参数！请检查微调策略设置。")
                #     print(f"检查到 {len(trainable_params)} 个可训练参数组")
                
                log_probs = self.model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                
                loss, loss_av, true_false = loss_nll(S, log_probs, mask * chain_M)
                
                self.optimizer.zero_grad()
                loss_av.backward()
                
                # if batch_idx == 0:
                #     grad_count = 0
                #     for name, param in self.model.named_parameters():
                #         if param.requires_grad and param.grad is not None:
                #             grad_count += 1
                #     print(f"有梯度的参数数量: {grad_count}")
                #     if grad_count == 0:
                #         print("警告: 没有参数收到梯度！")
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 
                    self.args.grad_clip
                )
                
                if hasattr(self.optimizer, 'step') and callable(self.optimizer.step):
                    self.optimizer.step()
                else:
                    self.optimizer.optimizer.step()
                
                if hasattr(self, 'scheduler') and self.args.optimizer != 'noam':
                    self.scheduler.step()
                
                valid_mask = mask * chain_M
                accuracy = calculate_accuracy(S, log_probs, valid_mask)
                perplexity = calculate_perplexity(log_probs, S, valid_mask)
                
                total_loss += loss_av.item()
                total_accuracy += accuracy
                total_perplexity += perplexity
                num_batches += 1
                
                current_lr = self.get_current_lr()
                progress_bar.set_postfix({
                    'Loss': f'{loss_av.item():.4f}',
                    'Acc': f'{accuracy:.4f}',
                    'PPL': f'{perplexity:.2f}',
                    'LR': f'{current_lr:.2e}'
                })
                
                if self.args.use_wandb:
                    wandb.log({
                        'train_loss_step': loss_av.item(),
                        'train_accuracy_step': accuracy,
                        'train_perplexity_step': perplexity,
                        'learning_rate': current_lr,
                        'step': batch_idx
                    })
                    
            except Exception as e:
                print(f"Error for {batch_idx} : {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches
            avg_perplexity = total_perplexity / num_batches
        else:
            avg_loss = float('inf')
            avg_accuracy = 0.0
            avg_perplexity = float('inf')
        
        return avg_loss, avg_accuracy, avg_perplexity
    
    def validate(self, dataloader):
        """Model validation"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_perplexity = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                try:
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
                        batch, self.device
                    )
                    
                    log_probs = self.model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask * chain_M)
                    
                    valid_mask = mask * chain_M
                    accuracy = calculate_accuracy(S, log_probs, valid_mask)
                    perplexity = calculate_perplexity(log_probs, S, valid_mask)
                    
                    total_loss += loss_av.item()
                    total_accuracy += accuracy
                    total_perplexity += perplexity
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error for Validation: {e}")
                    continue
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches
            avg_perplexity = total_perplexity / num_batches
        else:
            avg_loss = float('inf')
            avg_accuracy = 0.0
            avg_perplexity = float('inf')
        
        return avg_loss, avg_accuracy, avg_perplexity
    
    def get_current_lr(self):
        """Get current learning rate"""
        if hasattr(self.optimizer, 'param_groups'):
            return self.optimizer.param_groups[0]['lr']
        elif hasattr(self.optimizer, '_rate'):
            return self.optimizer._rate
        else:
            return self.args.learning_rate
    
    def save_checkpoint(self, epoch, train_loss, val_loss, train_acc, val_acc, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self.optimizer, 'state_dict') else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'args': self.args,
            'noise_level': self.args.augment_eps,
            'num_edges': self.args.k_neighbors
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.args.output_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.args.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
    
    def train(self):
        dataset = ThermophileDataset(self.args.data_file)
        
        # Dataset split
        # train_size = int(0.8 * len(dataset))
        # val_size = len(dataset) - train_size
        val_size = 1000
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=self.args.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=self.args.num_workers
        )
        
        print(f"Size of Train: {len(train_dataset)}")
        print(f"Size of Validation: {len(val_dataset)}")
        
        for epoch in range(self.args.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.args.num_epochs}")
            
            train_loss, train_acc, train_ppl = self.train_epoch(train_loader)
            
            val_loss, val_acc, val_ppl = self.validate(val_loader)
            
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, PPL: {train_ppl:.2f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, PPL: {val_ppl:.2f}")
            
            self.metrics['epoch'].append(epoch)
            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_acc'].append(val_acc)
            self.metrics['train_ppl'].append(train_ppl)
            self.metrics['val_ppl'].append(val_ppl)
            self.metrics['lr'].append(self.get_current_lr())
            
            self.save_metrics()

            if self.args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss_epoch': train_loss,
                    'train_accuracy_epoch': train_acc,
                    'train_perplexity_epoch': train_ppl,
                    'val_loss_epoch': val_loss,
                    'val_accuracy_epoch': val_acc,
                    'val_perplexity_epoch': val_ppl
                })
            
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"New best ACC: {val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, train_loss, val_loss, train_acc, val_acc, is_best)
            
            # Check early stop
            if self.args.early_stopping > 0 and self.patience_counter >= self.args.early_stopping:
                print(f"Early Stop, teminalate training at epoch {epoch+1}")
                break
            
            # check lr
            # if not is_best and self.patience_counter % 5 == 0 and self.patience_counter > 0:
            #     if hasattr(self.optimizer, 'param_groups'):
            #         for param_group in self.optimizer.param_groups:
            #             param_group['lr'] *= 0.5
            #         print(f"lr: {self.get_current_lr():.2e}")

    def save_metrics(self):
        """save metric to file for drawing"""
        metrics_file = os.path.join(self.args.output_dir, 'training_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='ProteinMPNN thermiphile fine-tuning')
    
    parser.add_argument('--data_file', type=str, default='./finetune/dataset56_pkl/splits/train_val_data.pkl',
                        help='path to training data')
    parser.add_argument('--pretrained_model', type=str, default='./vanilla_model_weights/v_48_030.pt',
                        help='path to pretrained model')
    

    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden layer dim')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='layer num of encoder')
    parser.add_argument('--num_decoder_layers', type=int, default=3, help='layer num of decoder')
    parser.add_argument('--augment_eps', type=float, default=0.02, help='value of noise level')
    parser.add_argument('--k_neighbors', type=int, default=48, help='num of neighbors')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    
    parser.add_argument('--finetune_strategy', type=str, default='full',
                        choices=['full', 'decoder_only', 'top_layers', 'feature_extractor_frozen', 'progressive'],
                        help='ft strategy')
    
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='num of train epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='lr')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay value')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clip value')
    parser.add_argument('--warmup_steps', type=int, default=5, help='warmup steps')
    
    parser.add_argument('--optimizer', type=str, default='adamw', 
                        choices=['adamw', 'noam'], help='optimizer type')
    parser.add_argument('--scheduler_restart', type=int, default=10, 
                        help='cosine annealing scheduler restart interval')
    
    parser.add_argument('--output_dir', type=str, default='./thermophile_finetune_output',
                        help='output dir')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers for data loader')
    parser.add_argument('--early_stopping', type=int, default=10, help='patience of early stopping, 0 means no early stopping')
    parser.add_argument('--use_wandb', action='store_true', help='whether to use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='proteinmpnn-thermophile',
                        help='wandb project name')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # init wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"traditional_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    trainer = ProteinMPNNTrainer(args)
    trainer.train()
    
    print("Train finished successfully!")


if __name__ == "__main__":
    main()