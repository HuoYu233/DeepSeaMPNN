import pickle

def simple_pkl_to_fasta(pkl_file, fasta_file):
    """
    简单版本的PKL转FASTA
    
    Args:
        pkl_file (str): 输入PKL文件路径
        fasta_file (str): 输出FASTA文件路径
    """
    try:
        # 读取PKL文件
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"读取到 {len(data)} 条记录")
        
        # 生成FASTA内容
        fasta_lines = []
        
        for i, record in enumerate(data):
            if i % 1000 == 0 and i > 0:
                print(f"已处理 {i} 条记录...")
            
            # 获取name和seq
            name = record.get('name', f'sequence_{i+1}')
            seq = record.get('seq', '')
            
            # 清理序列
            seq = ''.join(str(seq).split())
            
            if seq:  # 只添加非空序列
                fasta_lines.append(f">{name}")
                fasta_lines.append(seq)
        
        # 写入FASTA文件
        with open(fasta_file, 'w') as f:
            f.write("\n".join(fasta_lines))
        
        print(f"成功写入 {len(fasta_lines)//2} 条序列到 {fasta_file}")
        
    except Exception as e:
        print(f"转换失败: {e}")

# 使用示例
if __name__ == "__main__":
    simple_pkl_to_fasta("test_data.pkl", "input_test.fasta")