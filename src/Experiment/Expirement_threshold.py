from Naive_Clustering import Naive_clustering_Streaming
from Fixed_Reference_Clustering import Fixed_Reference_Clustering_Streaming
from Multiple_Reference_Clustering import Multiple_ref_clustering_streaming
from dataset.Slinding_window import process_streaming_data
import matplotlib

matplotlib.use('Agg')

if __name__ == '__main__':
    # 设置数据路径和窗口大小
    updates_path1 = 'dataset/Reddit'
    window_size1 = 500  # 例如，3天
    theta1 = 5.0
    theta2 = 10.0
    theta3 = 15.0
    theta4 = 20.0
    window_num = 1000
    para_theta = 1.0

    # Reddit数据集
    windows_update1_1, vectors1 = process_streaming_data(updates_path1, window_size1)

    # 实验3：相同window size，不同threshold，执行不同的方法在Reddit
    # 方法二：Naive
    time_reddit_size1_naive_t1 = Multiple_ref_clustering_streaming(windows_update1_1, vectors1, theta1, window_num, para_theta)
    time_reddit_size1_naive_t2 = Multiple_ref_clustering_streaming(windows_update1_1, vectors1, theta2, window_num, para_theta)
    time_reddit_size1_naive_t3 = Multiple_ref_clustering_streaming(windows_update1_1, vectors1, theta3, window_num, para_theta)
    time_reddit_size1_naive_t4 = Multiple_ref_clustering_streaming(windows_update1_1, vectors1, theta4, window_num, para_theta)


    # 方法三: Multiple Reference
    time_reddit_size1_mul_t1 = Multiple_ref_clustering_streaming(windows_update1_1, vectors1, theta1, window_num, para_theta)
    time_reddit_size1_mul_t2 = Multiple_ref_clustering_streaming(windows_update1_1, vectors1, theta2, window_num, para_theta)
    time_reddit_size1_mul_t3 = Multiple_ref_clustering_streaming(windows_update1_1, vectors1, theta3, window_num, para_theta)
    time_reddit_size1_mul_t4 = Multiple_ref_clustering_streaming(windows_update1_1, vectors1, theta4, window_num, para_theta)



    import matplotlib.pyplot as plt

    # 创建窗口索引，假设所有方法运行的窗口数量相同
    window_indices = list(range(len(time_reddit_size1_mul_t1)))

    # 绘制每种方法的执行时间
    plt.figure(figsize=(10, 6))  # 设置图形大小
    plt.plot(window_indices, time_reddit_size1_mul_t1, label='Threshold = 15', marker='o', linestyle='-')
    plt.plot(window_indices, time_reddit_size1_mul_t2, label='Threshold = 30', marker='s', linestyle='--')
    plt.plot(window_indices, time_reddit_size1_mul_t3, label='Threshold = 45', marker='^', linestyle='-.')
    plt.plot(window_indices, time_reddit_size1_mul_t4, label='Threshold = 60', marker='d', linestyle=':')

