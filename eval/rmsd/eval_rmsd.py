import os
from Bio.PDB import MMCIFParser, Superimposer

def find_all_cif_files(folder):
    """递归查找文件夹中所有的cif文件"""
    cif_files = []
    if not os.path.exists(folder):
        print(f"错误: 文件夹不存在 - {folder}")
        return cif_files
        
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.cif'):
                full_path = os.path.join(root, file)
                cif_files.append(full_path)
    return cif_files

def calculate_rmsd(folder_a, folder_b, atom_type='CA', min_length=100, top_k=5):
    """
    计算所有文件的RMSD，但在最后输出结果时，只筛选出长度 > min_length 的样本
    """
    
    parser = MMCIFParser(QUIET=True)
    superimposer = Superimposer()
    
    print(f"查找文件夹A: {folder_a}")
    cif_files_a = find_all_cif_files(folder_a)
    print(f"查找文件夹B: {folder_b}")
    cif_files_b = find_all_cif_files(folder_b)
    
    if not cif_files_a or not cif_files_b:
        print("错误: 未找到文件")
        return
    
    file_dict_a = {os.path.basename(f): f for f in cif_files_a}
    file_dict_b = {os.path.basename(f): f for f in cif_files_b}
    
    common_files = set(file_dict_a.keys()) & set(file_dict_b.keys())
    
    if not common_files:
        print("错误: 没有找到匹配的文件名")
        return
    
    print(f"找到 {len(common_files)} 对匹配文件，开始全部计算...")
    
    rmsd_records = []
    
    for cif_file in sorted(common_files):
        try:
            file_a = file_dict_a[cif_file]
            file_b = file_dict_b[cif_file]
            
            struct_a = parser.get_structure('A', file_a)
            struct_b = parser.get_structure('B', file_b)
            
            # 获取CA原子用于计算长度
            ca_atoms_a = [atom for atom in struct_a.get_atoms() if atom.get_name() == 'CA']
            seq_len = len(ca_atoms_a) # 记录长度
            
            # --- 即使长度不满足也继续计算 RMSD (不执行 continue) ---
            
            # 根据参数选择比对原子
            if atom_type.upper() == 'CA':
                atoms_a = ca_atoms_a
                atoms_b = [atom for atom in struct_b.get_atoms() if atom.get_name() == 'CA']
            elif atom_type.lower() == 'backbone':
                backbone_names = ['N', 'CA', 'C', 'O']
                atoms_a = [atom for atom in struct_a.get_atoms() if atom.get_name() in backbone_names]
                atoms_b = [atom for atom in struct_b.get_atoms() if atom.get_name() in backbone_names]
            elif atom_type.lower() == 'all':
                atoms_a = list(struct_a.get_atoms())
                atoms_b = list(struct_b.get_atoms())
            else:
                return
            
            if len(atoms_a) != len(atoms_b) or len(atoms_a) == 0:
                print(f"跳过 {cif_file}: 原子数量不匹配或为0")
                continue
            
            superimposer.set_atoms(atoms_a, atoms_b)
            rmsd = superimposer.rms
            
            # 记录所有结果，不管长度多少
            rmsd_records.append({'name': cif_file, 'rmsd': rmsd, 'len': seq_len})
            print(f"{cif_file}: RMSD = {rmsd:.3f} Å (长度: {seq_len})")
            
        except Exception as e:
            print(f"出错 {cif_file}: {e}")
    
    # === 结果处理阶段 ===
    if rmsd_records:
        # 在这里进行过滤：只选出长度 > min_length 的
        filtered_records = [r for r in rmsd_records if r['len'] > min_length]
        
        # 按照 RMSD 从小到大排序
        filtered_records.sort(key=lambda x: x['rmsd'])
        
        print(f"\n=== 结果汇总 (仅展示长度 > {min_length} 的样本) ===")
        print(f"总计算文件数: {len(rmsd_records)}")
        print(f"符合长度条件的文件数: {len(filtered_records)}")
        
        if filtered_records:
            avg_rmsd = sum(r['rmsd'] for r in filtered_records) / len(filtered_records)
            print(f"符合条件的平均 RMSD: {avg_rmsd:.3f} Å")
            
            # 截取前 K 个
            top_k_records = filtered_records[:top_k]
            
            print(f"\n>>> 表现最好 (RMSD最小) 的 Top {len(top_k_records)} <<<")
            print(f"{'文件名':<35} {'RMSD':<10} {'长度':<10}")
            print("-" * 55)
            for rec in top_k_records:
                print(f"{rec['name']:<35} {rec['rmsd']:.3f} Å    {rec['len']}")
        else:
            print(f"警告: 虽然计算了 {len(rmsd_records)} 个文件，但没有一个长度大于 {min_length}。")
            
    else:
        print("没有成功计算任何文件的RMSD")

if __name__ == "__main__":
    
    atom_type = 'CA' 
    
    folder_a = "/data2/luozheng/ProteinMPNN/finetune/dataset56_testdata"
    folder_b = "/data2/luozheng/ProteinMPNN/finetune/dataset56_test_generated_cif"

    calculate_rmsd(folder_a, folder_b, atom_type, min_length=100, top_k=5)