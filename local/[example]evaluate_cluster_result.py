import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics.cluster import contingency_matrix
from collections import Counter
from scipy.optimize import linear_sum_assignment

def purity_score(y_true, y_pred):
    """计算聚类纯度 (Purity)"""
    contingency = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

def clustering_accuracy(y_true, y_pred):
    """
    计算聚类准确率 (Clustering Accuracy)
    使用匈牙利算法寻找最优的类别映射
    """
    contingency = contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    accuracy = contingency[row_ind, col_ind].sum() / np.sum(contingency)
    return accuracy

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估说话人聚类结果')
    parser.add_argument('--workspace', default='.', type=str, help='工作目录路径')
    args = parser.parse_args()
    
    # 使用参数指定的工作目录
    workspace = args.workspace
    
    # 创建结果目录（如果不存在）
    os.makedirs(os.path.join(workspace, 'result'), exist_ok=True)
    
    # 读取聚类结果和真实标注
    cluster_result_path = os.path.join(workspace, 'result/cluster_result.csv')
    ground_truth_path = os.path.join(workspace, 'dataset/wav_metadata.csv')
    
    # 检查文件是否存在
    if not os.path.exists(cluster_result_path):
        print(f"错误: 聚类结果文件不存在: {cluster_result_path}")
        return
    
    if not os.path.exists(ground_truth_path):
        print(f"错误: 真实标签文件不存在: {ground_truth_path}")
        return
    
    cluster_df = pd.read_csv(cluster_result_path)
    ground_truth_df = pd.read_csv(ground_truth_path)
    
    # 合并数据集，确保比较相同的音频文件
    merged_df = pd.merge(cluster_df, ground_truth_df, on='wav_name')
    
    # 提取预测标签和真实标签
    y_pred = merged_df['speaker_id_pred'].values
    y_true = merged_df['speaker_id_true'].values
    
    # 计算评估指标
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    homogeneity = homogeneity_score(y_true, y_pred)
    completeness = completeness_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)
    accuracy = clustering_accuracy(y_true, y_pred)
    
    # 统计真实说话人和聚类说话人的分布
    true_speaker_counts = Counter(y_true)
    pred_speaker_counts = Counter(y_pred)
    
    # 将评估结果写入文件
    output_path = os.path.join(workspace, 'result/cluster_result_eval.txt')
    with open(output_path, 'w') as f:
        f.write("说话人聚类评估结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("基本统计信息:\n")
        f.write(f"总样本数: {len(merged_df)}\n")
        f.write(f"真实说话人数量: {len(true_speaker_counts)}\n")
        f.write(f"聚类得到的说话人数量: {len(pred_speaker_counts)}\n\n")
        
        # 其余代码保持不变...
        f.write("真实说话人分布:\n")
        for speaker_id, count in sorted(true_speaker_counts.items()):
            f.write(f"说话人 {speaker_id}: {count} 个样本\n")
        f.write("\n")
        
        f.write("聚类结果说话人分布:\n")
        for speaker_id, count in sorted(pred_speaker_counts.items()):
            f.write(f"聚类 {speaker_id}: {count} 个样本\n")
        f.write("\n")
        
        f.write("评估指标:\n")
        
        f.write("调整兰德指数 (ARI): 测量两个聚类结果的相似度，考虑了偶然分组的影响\n")
        f.write("取值范围: [-1, 1], 1表示完美匹配, 0表示随机分配, 负值表示比随机更差 [越高越好]\n")
        f.write(f"调整兰德指数 (ARI): {ari:.4f}\n\n")
        
        f.write("归一化互信息 (NMI): 衡量聚类结果与真实标签之间共享的信息量\n")
        f.write("取值范围: [0, 1], 1表示完美匹配, 对聚类数量不敏感 [越高越好]\n")
        f.write(f"归一化互信息 (NMI): {nmi:.4f}\n\n")
        
        f.write("同质性 (Homogeneity): 衡量每个聚类是否只包含单一类别的样本\n")
        f.write("取值范围: [0, 1], 1表示每个聚类只包含一个类别的样本 [越高越好]\n")
        f.write(f"同质性 (Homogeneity): {homogeneity:.4f}\n\n")
        
        f.write("完整性 (Completeness): 衡量同一类别的所有样本是否都被分到同一聚类\n")
        f.write("取值范围: [0, 1], 1表示同一类别的样本都在同一聚类中 [越高越好]\n")
        f.write(f"完整性 (Completeness): {completeness:.4f}\n\n")
        
        f.write("V-measure: 同质性和完整性的调和平均值，平衡了这两个互补指标\n")
        f.write("取值范围: [0, 1], 不受类别数量变化影响 [越高越好]\n")
        f.write(f"V-measure: {v_measure:.4f}\n\n")
        
        f.write("聚类纯度 (Purity): 测量每个聚类中占主导地位的类别的纯度\n") 
        f.write("取值范围: [0, 1], 1表示每个聚类中只有一个类别，但不惩罚过度聚类 [越高越好]\n")
        f.write(f"聚类纯度 (Purity): {purity:.4f}\n\n")
        
        f.write("聚类准确率 (Clustering Accuracy): 使用匈牙利算法寻找最优的标签映射后计算准确率\n")
        f.write("取值范围: [0, 1], 能够处理聚类数与真实类别数不一致的情况 [越高越好]\n")
        f.write(f"聚类准确率 (Clustering Accuracy): {accuracy:.4f}\n")
    
    # 打印评估结果
    print("评估结果:")
    print("=" * 50)
    print(f"总样本数: {len(merged_df)}")
    print(f"真实说话人数量: {len(true_speaker_counts)}")
    print(f"聚类得到的说话人数量: {len(pred_speaker_counts)}")
    
    print("\nARI:", ari)
    print("NMI:", nmi)
    print("Homogeneity:", homogeneity)
    print("Completeness:", completeness)
    print("V-measure:", v_measure)
    print("Purity:", purity)
    print("Clustering Accuracy:", accuracy)
    
    print(f"\n评估结果已写入: {output_path}")

if __name__ == "__main__":
    main()