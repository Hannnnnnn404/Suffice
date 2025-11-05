import random

import numpy as np

from Multiple_Reference_Clustering import Multiple_ref_clustering_streaming_parameters
import matplotlib
from dataset.Slinding_window import process_streaming_data

matplotlib.use('Agg')

if __name__ == '__main__':
    # 设置数据路径和窗口大小
    updates_path1 = 'dataset/Reddit'
    window_size1 = 500  # 例如，3天

    theta = 30.0
    window_num = 1000
    para_theta = 10.0

    windows_update, vector = process_streaming_data(updates_path1, window_size1)

    time_reddit_size1_mul_t1_ht1, _ = Multiple_ref_clustering_streaming_parameters(windows_update, vector, theta,
                                                                                   window_num,
                                                                                   para_theta, 10, 18, 10)
    time_reddit_size1_mul_t2_ht1, _ = Multiple_ref_clustering_streaming_parameters(windows_update, vector, theta,
                                                                                   window_num,
                                                                                   para_theta, 20, 18, 10)
    time_reddit_size1_mul_t3_ht1, _ = Multiple_ref_clustering_streaming_parameters(windows_update, vector, theta,
                                                                                   window_num,
                                                                                   para_theta, 50, 18, 10)
    time_reddit_size1_mul_t4_ht1, _ = Multiple_ref_clustering_streaming_parameters(windows_update, vector, theta,
                                                                                   window_num,
                                                                                   para_theta, 100, 18, 10)

    # Mul方法的累计时间
    sum_list_ht1 = [time_reddit_size1_mul_t1_ht1, time_reddit_size1_mul_t2_ht1, time_reddit_size1_mul_t3_ht1,
                    time_reddit_size1_mul_t4_ht1]
    sum_mul_ht1 = []
    for a_list in sum_list_ht1:
        s = sum(a_list) / len(a_list)
        sum_mul_ht1.append(s)

    print("ht1", sum_mul_ht1)

    # 方法三: Multiple Reference
    time_reddit_size1_mul_t1_ht2, _ = Multiple_ref_clustering_streaming_parameters(windows_update, vector, theta,
                                                                                   window_num,
                                                                                   para_theta, 10, 10, 10)
    time_reddit_size1_mul_t2_ht2, _ = Multiple_ref_clustering_streaming_parameters(windows_update, vector, theta,
                                                                                   window_num,
                                                                                   para_theta, 10, 20, 10)
    time_reddit_size1_mul_t3_ht2, _ = Multiple_ref_clustering_streaming_parameters(windows_update, vector, theta,
                                                                                   window_num,
                                                                                   para_theta, 10, 50, 10)
    time_reddit_size1_mul_t4_ht2, _ = Multiple_ref_clustering_streaming_parameters(windows_update, vector, theta,
                                                                                   window_num,
                                                                                   para_theta, 10, 100, 10)

    # Mul方法的累计时间
    sum_list_ht2 = [time_reddit_size1_mul_t1_ht2, time_reddit_size1_mul_t2_ht2, time_reddit_size1_mul_t3_ht2,
                    time_reddit_size1_mul_t4_ht2]
    sum_mul_ht2 = []
    for a_list in sum_list_ht2:
        s = sum(a_list) / len(a_list)
        sum_mul_ht2.append(s)

    print("ht2", sum_mul_ht2)
