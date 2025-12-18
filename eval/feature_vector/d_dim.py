# import numpy as np

# obj = np.load('feature_vector/origin.npy', allow_pickle=True)
# print(obj.shape)
# print(obj[0])
# print('=' * 40)
# print(obj[1])

# print(obj[-1].shape, obj[-2].shape)
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
def load_and_process_data(file1_path, file2_path):
    """加载两个npy文件并返回处理后的数据"""
    print("正在加载数据...")
    
    data1 = np.load(file1_path, allow_pickle=True)
    data2 = np.load(file2_path, allow_pickle=True)
    
    print(f"文件1样本数: {len(data1)}")
    print(f"文件2样本数: {len(data2)}")
    print(f"文件1示例形状: {data1[0].shape}")
    print(f"文件2示例形状: {data2[0].shape}")
    
    return data1, data2

def aggregate_features(data, method='mean'):
    """对数据进行池化聚合"""
    if len(data.shape) == 2:  # 已经是 [样本数, 特征维度]
        return data
    elif len(data.shape) == 3:
        aggregated_vectors = []
        
        for vec in data:
            if method == 'mean':
                # 全局平均池化
                aggregated_vec = np.mean(vec, axis=0)
            elif method == 'max':
                # 全局最大池化
                aggregated_vec = np.max(vec, axis=0)
            else:
                raise ValueError("池化方法必须是 'mean' 或 'max'")
            
            aggregated_vectors.append(aggregated_vec)
        
        return np.array(aggregated_vectors)

def reduce_dimension(vectors, method='umap'):
    """降维到2维"""
    print(f"使用{method.upper()}进行降维...")
    
    if method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(vectors)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embedding = reducer.fit_transform(vectors)
    else:
        raise ValueError("降维方法必须是 'umap' 或 'tsne'")
    
    return embedding

def plot_comparison(embeddings1, embeddings2, labels, save_path):
    """绘制对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 定义颜色
    color1 = '#1f77b4'  # 蓝色
    color2 = '#ff7f0e'  # 橙色
    
    for i, (ax, method) in enumerate(zip(axes, ['Mean Pooling', 'Max Pooling'])):
        emb1 = embeddings1[i]
        emb2 = embeddings2[i]
        
        # 绘制散点图
        ax.scatter(emb1[:, 0], emb1[:, 1], c=color1, alpha=0.7, s=30, 
                  label=labels[0], edgecolors='white', linewidth=0.5)
        ax.scatter(emb2[:, 0], emb2[:, 1], c=color2, alpha=0.7, s=30, 
                  label=labels[1], edgecolors='white', linewidth=0.5)
        
        ax.set_title(f'{method} - UMAP Visualizaion', fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存: {save_path}")
    plt.show()

def main(file1_path, file2_path, label1='通用型', label2='耐热型'):
    """主函数"""
    # 1. 加载数据
    data1, data2 = load_and_process_data(file1_path, file2_path)
    
    # 2. 分别进行两种池化
    print("\n正在进行特征聚合...")
    
    # 平均池化
    mean_pooled1 = aggregate_features(data1, 'mean')
    mean_pooled2 = aggregate_features(data2, 'mean')
    
    # 最大池化
    max_pooled1 = aggregate_features(data1, 'max')
    max_pooled2 = aggregate_features(data2, 'max')
    
    print(f"平均池化后形状 - {label1}: {mean_pooled1.shape}")
    print(f"平均池化后形状 - {label2}: {mean_pooled2.shape}")
    print(f"最大池化后形状 - {label1}: {max_pooled1.shape}")
    print(f"最大池化后形状 - {label2}: {max_pooled2.shape}")
    
    # 3. 合并数据进行降维（确保在同一空间）
    print("\n正在进行降维...")
    
    # 合并所有数据进行统一的降维
    all_mean_pooled = np.vstack([mean_pooled1, mean_pooled2])
    all_max_pooled = np.vstack([max_pooled1, max_pooled2])
    
    # 降维
    mean_embedding_all = reduce_dimension(all_mean_pooled, 'umap')
    max_embedding_all = reduce_dimension(all_max_pooled, 'umap')
    
    # 分割回原来的两组数据
    n1 = len(mean_pooled1)
    mean_embedding1 = mean_embedding_all[:n1]
    mean_embedding2 = mean_embedding_all[n1:]
    
    max_embedding1 = max_embedding_all[:n1]
    max_embedding2 = max_embedding_all[n1:]
    
    # 4. 绘制对比图
    print("\n正在生成可视化图像...")
    
    embeddings1 = [mean_embedding1, max_embedding1]
    embeddings2 = [mean_embedding2, max_embedding2]
    labels = [label1, label2]
    
    # 生成保存路径
    base_name = f"{label1}_vs_{label2}"
    save_path = f"{base_name}_comparison.png"
    
    plot_comparison(embeddings1, embeddings2, labels, save_path)
    
    # 5. 保存降维后的坐标（可选）
    np.save(f'{base_name}_mean_pooled_coords.npy', {
        f'{label1}': mean_embedding1,
        f'{label2}': mean_embedding2
    })
    np.save(f'{base_name}_max_pooled_coords.npy', {
        f'{label1}': max_embedding1,
        f'{label2}': max_embedding2
    })
    print(f"降维坐标已保存为: {base_name}_*_pooled_coords.npy")

if __name__ == "__main__":
    file1_path = "original.npy"
    file2_path = "thermal.npy"
    label1 = "Original"
    label2 = "Thermal"
    
    main(file1_path, file2_path, label1, label2)