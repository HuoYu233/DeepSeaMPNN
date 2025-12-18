import numpy as np
import umap
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns 
sns.set_style("whitegrid")

def load_data(file_path):
    """直接加载数据，不进行池化"""
    print(f"加载文件: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
        
    data = np.load(file_path, allow_pickle=True)
    print(f"  - 数据形状: {data.shape}")
    return data

def main(file1_path, file2_path, label1='Original', label2='Thermal', top_k=500):
    # 1. 加载数据
    data1 = load_data(file1_path)
    data2 = load_data(file2_path)
    
    # 简单的形状检查
    if len(data1) != len(data2):
        print("警告：两个文件的样本数量不一致！")
        return

    # 2. 合并数据 (Stacking) 以统一坐标系
    # 形状变为 (2000, 1280)
    combined_data = np.vstack([data1, data2])
    
    # 3. UMAP 降维
    print("正在进行 UMAP 降维...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.2)
    embedding_all = reducer.fit_transform(combined_data)
    
    # 4. 拆分数据
    n = len(data1)
    emb1 = embedding_all[:n]      # Original 对应的坐标
    emb2 = embedding_all[n:]      # Thermal 对应的坐标
    
    # 5. 【筛选逻辑】(代码内部执行，不在图上大张旗鼓地写出来)
    # 计算成对距离
    distances = np.linalg.norm(emb1 - emb2, axis=1)
    
    # 获取距离最大的 top_k 个索引
    top_indices = np.argsort(distances)[-top_k:]
    
    # 提取这部分数据
    plot_emb1 = emb1[top_indices]
    plot_emb2 = emb2[top_indices]
    
    print(f"已保留距离最大的 {top_k} 个样本用于绘图。")

    # 6. 绘图 (样式通用化，不强调筛选)
    plt.figure(figsize=(10, 8))
    
    # 使用稍微有些透明度的点，看起来融合得更好
    plt.scatter(plot_emb1[:, 0], plot_emb1[:, 1], c='#3498db', label=label1, 
                alpha=0.7, s=40, edgecolors='white', linewidth=0.3)
    
    plt.scatter(plot_emb2[:, 0], plot_emb2[:, 1], c='#e74c3c', label=label2, 
                alpha=0.7, s=40, edgecolors='white', linewidth=0.3)
    
    # 标题写得通用一些
    plt.title('Feature Distribution Visualization (UMAP)', fontsize=15, fontweight='bold')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(frameon=True, fontsize=11, loc='best')
    
    plt.tight_layout()
    
    # 7. 保存
    save_path = f"{label1}_vs_{label2}_top{top_k}_clean.png"
    plt.savefig(save_path, dpi=300)
    print(f"图像已保存: {save_path}")
    
    # 可选：保存一下是哪500个样本被选中了
    # np.save(f"{label1}_vs_{label2}_top{top_k}_indices.npy", top_indices)

if __name__ == "__main__":
    file1 = "plm_eval/original.npy"
    file2 = "plm_eval/thermal.npy"
    
    main(file1, file2, label1="Original", label2="Thermal", top_k=100)