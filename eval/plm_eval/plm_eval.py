import torch
import esm
import numpy as np
from Bio import SeqIO
from typing import List, Tuple
import time
import os

class FASTAEmbeddingGenerator:
    def __init__(self, model_name: str = "esm2_t36_3B_UR50D", device: str = None):
        """
        初始化ESM模型
        
        Args:
            model_name: ESM模型名称，使用以下之一:
                - 'esm2_t6_8M_UR50D' (8M参数，最快)
                - 'esm2_t12_35M_UR50D' (35M参数)
                - 'esm2_t30_150M_UR50D' (150M参数)
                - 'esm2_t33_650M_UR50D' (650M参数，推荐)
                - 'esm2_t36_3B_UR50D' (3B参数)
                - 'esm2_t48_15B_UR50D' (15B参数)
            device: 计算设备 ('cuda' 或 'cpu')，默认自动选择
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        print(f"加载模型: {model_name}...")
        
        model_loaders = {
            "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
            "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
            "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
            "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
            "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D,
            "esm2_t48_15B_UR50D": esm.pretrained.esm2_t48_15B_UR50D,
        }
        
        if model_name in model_loaders:
            self.model, self.alphabet = model_loaders[model_name]()
        else:
            print(f"模型 {model_name} 不在预定义列表中，尝试从Hub加载...")
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.batch_converter = self.alphabet.get_batch_converter()
        
        print(f"模型加载完成! 嵌入维度: {self.model.embed_dim}")
        
    def read_fasta_file(self, fasta_path: str) -> List[Tuple[str, str]]:
        """
        读取FASTA文件，返回序列列表
        
        Args:
            fasta_path: FASTA文件路径
            
        Returns:
            列表，每个元素为 (序列ID, 序列字符串)
        """
        sequences = []
        print(f"读取FASTA文件: {fasta_path}")
        
        if not os.path.exists(fasta_path):
            print(f"错误: 文件 {fasta_path} 不存在!")
            return sequences
        
        try:
            with open(fasta_path, 'r') as f:
                for i, record in enumerate(SeqIO.parse(f, "fasta")):
                    seq_id = record.id
                    seq_str = str(record.seq).upper()
                    
                    # 过滤无效字符
                    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
                    seq_clean = ''.join([aa for aa in seq_str if aa in valid_aas])
                    
                    if len(seq_clean) > 0:
                        sequences.append((seq_id, seq_clean))
                    else:
                        print(f"警告: 序列 {seq_id} 不包含有效氨基酸，已跳过")
                    
                    # 只读取前1000条（如果文件很大）
                    if len(sequences) >= 1000:
                        print(f"已读取1000条序列，停止读取")
                        break
                        
        except Exception as e:
            print(f"读取FASTA文件时出错: {e}")
        
        print(f"读取到 {len(sequences)} 条有效序列")
        return sequences
    
    def process_batch(self, batch_sequences: List[Tuple[str, str]]) -> np.ndarray:
        """
        处理一批序列，返回嵌入向量
        
        Args:
            batch_sequences: 批序列列表 [(id1, seq1), (id2, seq2), ...]
            
        Returns:
            嵌入向量数组 [batch_size, embedding_dim]
        """
        if not batch_sequences:
            return np.array([])
        
        try:
            # 转换为模型输入格式
            batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_sequences)
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                # 获取最后一层的表示
                results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
                token_embeddings = results["representations"][self.model.num_layers]
                
                # 使用平均池化获取序列表示（更稳定）
                # 排除特殊token [CLS]和[EOS]
                sequence_lengths = (batch_tokens != self.alphabet.padding_idx).sum(dim=1)
                
                # 计算每个序列的有效token的平均
                embeddings_list = []
                for i in range(len(batch_sequences)):
                    # 获取非padding的token
                    valid_tokens = token_embeddings[i, 1:sequence_lengths[i]-1, :]  # 排除CLS和EOS
                    if len(valid_tokens) > 0:
                        seq_embedding = valid_tokens.mean(dim=0)
                    else:
                        # 如果序列太短，使用CLS token
                        seq_embedding = token_embeddings[i, 0, :]
                    embeddings_list.append(seq_embedding.cpu().numpy())
                
                return np.array(embeddings_list)
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU内存不足! 尝试减小batch_size")
            raise e
    
    def generate_embeddings(self, sequences: List[Tuple[str, str]], 
                           batch_size: int = 8) -> Tuple[List[str], np.ndarray]:
        """
        为所有序列生成嵌入
        
        Args:
            sequences: 序列列表 [(id1, seq1), (id2, seq2), ...]
            batch_size: 批处理大小
            
        Returns:
            (序列ID列表, 嵌入向量数组)
        """
        all_embeddings = []
        sequence_ids = []
        
        total_sequences = len(sequences)
        print(f"开始处理 {total_sequences} 条序列...")
        
        start_time = time.time()
        
        for i in range(0, total_sequences, batch_size):
            batch_end = min(i + batch_size, total_sequences)
            batch_sequences = sequences[i:batch_end]
            
            # 提取序列ID
            batch_ids = [seq[0] for seq in batch_sequences]
            sequence_ids.extend(batch_ids)
            
            # 处理当前批次
            batch_embeddings = self.process_batch(batch_sequences)
            
            if len(batch_embeddings) > 0:
                all_embeddings.append(batch_embeddings)
            
            # 显示进度
            processed = min(i + batch_size, total_sequences)
            if (i // batch_size) % 10 == 0 or processed == total_sequences:
                elapsed_time = time.time() - start_time
                if processed > 0:
                    avg_time_per_seq = elapsed_time / processed
                    remaining_time = avg_time_per_seq * (total_sequences - processed)
                    print(f"进度: {processed}/{total_sequences} | "
                          f"已用时: {elapsed_time:.1f}s | "
                          f"预计剩余: {remaining_time:.1f}s")
        
        # 合并所有批次的嵌入
        if all_embeddings:
            all_embeddings_np = np.vstack(all_embeddings)
        else:
            all_embeddings_np = np.array([])
        
        total_time = time.time() - start_time
        print(f"处理完成! 总用时: {total_time:.2f}秒")
        print(f"平均每条序列: {total_time/max(1, total_sequences):.3f}秒")
        print(f"生成的嵌入维度: {all_embeddings_np.shape}")
        
        return sequence_ids, all_embeddings_np
    
    def save_embeddings(self, sequence_ids: List[str], embeddings: np.ndarray, 
                       output_npy: str, output_ids: str = None):
        """
        保存嵌入向量到npy文件
        
        Args:
            sequence_ids: 序列ID列表
            embeddings: 嵌入向量数组
            output_npy: 输出npy文件路径
            output_ids: 输出序列ID文件路径（可选）
        """
        # 保存嵌入向量
        np.save(output_npy, embeddings)
        print(f"嵌入向量已保存到: {output_npy}")
        print(f"嵌入向量形状: {embeddings.shape}")
        print(f"数据类型: {embeddings.dtype}")
        
        # 保存序列ID（可选）
        # if output_ids:
        #     with open(output_ids, 'w') as f:
        #         for seq_id in sequence_ids:
        #             f.write(f"{seq_id}\n")
        #     print(f"序列ID已保存到: {output_ids}")
    
    def process_fasta_file(self, fasta_path: str, output_npy: str, 
                          batch_size: int = 8):
        """
        处理单个FASTA文件的完整流程
        
        Args:
            fasta_path: 输入FASTA文件路径
            output_npy: 输出npy文件路径
            batch_size: 批处理大小
        """
        print(f"\n{'='*60}")
        print(f"处理文件: {fasta_path}")
        print(f"{'='*60}")
        
        # 1. 读取FASTA文件
        sequences = self.read_fasta_file(fasta_path)
        
        if len(sequences) == 0:
            print("错误: 没有读取到有效序列!")
            return
        
        # 2. 生成嵌入
        sequence_ids, embeddings = self.generate_embeddings(sequences, batch_size)
        
        # 3. 保存结果
        base_name = os.path.splitext(output_npy)[0]
        ids_file = f"{base_name}_ids.txt"
        self.save_embeddings(sequence_ids, embeddings, output_npy, ids_file)


def main():
    """主函数：处理两个FASTA文件"""
    FASTA_FILE_1 = "thermal.fasta"  
    FASTA_FILE_2 = "original.fasta"
    
    OUTPUT_NPY_1 = "thermal.npy"
    OUTPUT_NPY_2 = "original.npy"
    
    MODEL_NAME = "esm2_t12_35M_UR50D"
    
    BATCH_SIZE = 16 
    
    DEVICE = 'cuda:1'

    print("初始化ESM模型...")
    generator = FASTAEmbeddingGenerator(
        model_name=MODEL_NAME,
        device=DEVICE
    )
    
    # 处理第一个文件
    generator.process_fasta_file(
        fasta_path=FASTA_FILE_1,
        output_npy=OUTPUT_NPY_1,
        batch_size=BATCH_SIZE
    )
    
    # 处理第二个文件
    generator.process_fasta_file(
        fasta_path=FASTA_FILE_2,
        output_npy=OUTPUT_NPY_2,
        batch_size=BATCH_SIZE
    )
    
    print("\n" + "="*60)
    print("处理完成!")
    print(f"文件1嵌入: {OUTPUT_NPY_1}")
    print(f"文件2嵌入: {OUTPUT_NPY_2}")
    print("="*60)


if __name__ == "__main__":
    main()