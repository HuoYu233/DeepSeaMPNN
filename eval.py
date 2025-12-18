import os
import glob
import re
import math

def quick_seq_recovery_stats(folder_path):
    """
    快速计算seq_recovery和perplexity统计
    """
    fa_files = glob.glob(os.path.join(folder_path, "*.fa"))
    
    recoveries = []
    perplexities = []
    
    for fa_file in fa_files:
        with open(fa_file, 'r') as f:
            content = f.read()
            # 提取seq_recovery
            recovery_matches = re.findall(r'seq_recovery=([\d.]+)', content)
            # 提取第二个global_score（在T=...后面的那个）
            global_score_matches = re.findall(r'>T=.*?global_score=([\d.]+)', content)
            # global_score_matches = re.findall(r'^>.*?global_score=([\d.]+)', content, re.MULTILINE)
            
            # 处理seq_recovery
            for match in recovery_matches:
                try:
                    recoveries.append(float(match))
                except ValueError:
                    continue
            
            # 处理global_score并计算perplexity
            for score_match in global_score_matches:
                try:
                    global_score = float(score_match)
                    perplexity = math.exp(global_score)
                    perplexities.append(perplexity)
                except ValueError:
                    continue
    
    print("=== 序列恢复率统计 ===")
    if recoveries:
        avg_recovery = sum(recoveries) / len(recoveries)
        print(f"  总序列数: {len(recoveries)}")
        print(f"  平均恢复率: {avg_recovery:.4f} ({avg_recovery*100:.2f}%)")
        print(f"  范围: [{min(recoveries):.4f}, {max(recoveries):.4f}]")
    else:
        print("  未找到seq_recovery数据")
        avg_recovery = 0
    
    print("\n=== 困惑度统计 ===")
    if perplexities:
        avg_perplexity = sum(perplexities) / len(perplexities)
        # 同时计算global_score的平均值，用于参考
        avg_global_score = sum([math.log(p) for p in perplexities]) / len(perplexities) if perplexities else 0
        
        print(f"  总序列数: {len(perplexities)}")
        print(f"  平均困惑度: {avg_perplexity:.4f}")
        print(f"  平均global_score: {avg_global_score:.4f}")
        print(f"  困惑度范围: [{min(perplexities):.4f}, {max(perplexities):.4f}]")
        return avg_recovery, avg_perplexity
    else:
        print("  未找到global_score数据")
        return avg_recovery, 0

folder_path = "../v_thermal_002_seq/seqs"
average_recovery, average_perplexity = quick_seq_recovery_stats(folder_path)