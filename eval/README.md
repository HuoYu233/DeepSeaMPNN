# 评估
## feature_vetctor
取模型最后一层向量，进行最大池化或平均池化，UAMP降维，观察聚类效果
origin.npy: 测试集经过原始MPNN的向量
thermal.npy: 测试集经过微调MPNN的向量

## plm_eval
序列经过ESM2的嵌入向量，UMAP降维，观察聚类效果
esmX_dim: 经过x层嵌入得到dim维向量的相关数据
d_dim_filter: 选取差距最大的k个点进行降维聚类
plm_eval: 使用plm处理序列得到嵌入向量
