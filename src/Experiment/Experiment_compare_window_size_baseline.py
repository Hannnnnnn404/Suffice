from Links import LinksStream
from HSRC import HSRCStream
from DEN_Stream import DENStream
from D_Stream import DStream
from Naive_Clustering import Naive_clustering_Streaming
from Fixed_Reference_Clustering import Fixed_Reference_Clustering_Streaming
from Multiple_Reference_Clustering import Multiple_ref_clustering_streaming
from dataset.Slinding_window import process_streaming_data, process_streaming_data_over
from Fix_Dynamic import Fixed_Reference_Clustering_Streaming_Adaptive
from Multiple_Reference_Clustering_Expire import Multiple_ref_clustering_streaming_Expire

if __name__ == '__main__':
    # 设置数据路径和窗口大小
    updates_path1 = 'dataset/Reddit'
    window_size1 = 10  # 例如，3天
    window_size2 = 100
    window_size3 = 1000
    window_size4 = 5000
    window_size5 = 10000
    # [5, 10, 20, 50, 100]
    # [10, 100, 1000, 5000, 10000]

    theta = 30.0
    window_num = 1000
    para_theta = 10.0

    windows_update1_1, vector1_1 = process_streaming_data(updates_path1, window_size1)
    windows_update1_2, vector1_2 = process_streaming_data(updates_path1, window_size2)
    windows_update1_3, vector1_3 = process_streaming_data_over(updates_path1, window_size3, window_num)
    windows_update1_4, vector1_4 = process_streaming_data_over(updates_path1, window_size4, window_num)
    windows_update1_5, vector1_5 = process_streaming_data_over(updates_path1, window_size5, window_num)

    # Links
    time_reddit_size1_links, initial_time1 = LinksStream(windows_update1_1, vector1_1, Tc=1.0, Ts=1.0, window_num=window_num)
    time_reddit_size2_links, initial_time2 = LinksStream(windows_update1_2, vector1_2, Tc=1.0, Ts=1.0, window_num=window_num)
    time_reddit_size3_links, initial_time3 = LinksStream(windows_update1_3, vector1_3, Tc=1.0, Ts=1.0, window_num=window_num)
    time_reddit_size4_links, initial_time4 = LinksStream(windows_update1_4, vector1_4, Tc=1.0, Ts=1.0, window_num=window_num)
    time_reddit_size5_links, initial_time5 = LinksStream(windows_update1_5, vector1_5, Tc=1.0, Ts=1.0, window_num=window_num)

    # HSRC
    time_reddit_size1_hsrc, _ = HSRCStream(windows_updates=windows_update1_1, vectors=vector1_1, theta=theta, k=10, s=0.2,
                                           window_num=window_num, dim=len(vector1_1), device='cpu')
    time_reddit_size2_hsrc, _ = HSRCStream(windows_updates=windows_update1_2, vectors=vector1_2, theta=theta, k=10, s=0.2,
                                           window_num=window_num, dim=len(vector1_2), device='cpu')
    time_reddit_size3_hsrc, _ = HSRCStream(windows_updates=windows_update1_3, vectors=vector1_3, theta=theta, k=10, s=0.2,
                                           window_num=window_num, dim=len(vector1_3), device='cpu')
    time_reddit_size4_hsrc, _ = HSRCStream(windows_updates=windows_update1_4, vectors=vector1_4, theta=theta, k=10, s=0.2,
                                           window_num=window_num, dim=len(vector1_4), device='cpu')
    time_reddit_size5_hsrc, _ = HSRCStream(windows_updates=windows_update1_5, vectors=vector1_5, theta=theta, k=10, s=0.2,
                                           window_num=window_num, dim=len(vector1_5), device='cpu')

    # links的曲线
    average_list_links = [sum(time_reddit_size1_links) + initial_time1,
                          sum(time_reddit_size2_links) + initial_time2,
                          sum(time_reddit_size3_links) + initial_time3,
                          sum(time_reddit_size4_links) + initial_time4,
                          sum(time_reddit_size5_links) + initial_time5]
    average_window_size_links = []
    for a_list in average_list_links:
        avg = a_list / len(a_list)
        average_window_size_links.append(avg)

    # hsrc的曲线
    average_list_hsrc = [time_reddit_size1_hsrc, time_reddit_size2_hsrc, time_reddit_size3_hsrc,
                         time_reddit_size4_hsrc,
                         time_reddit_size5_hsrc]
    average_window_size_hsrc = []
    for a_list in average_list_hsrc:
        avg = sum(a_list) / len(a_list)
        average_window_size_hsrc.append(avg)

    print(average_window_size_links)
    print(average_window_size_hsrc)

    # Fix Adaptive
    time_reddit_size1_fix_adaptive, _ = Fixed_Reference_Clustering_Streaming_Adaptive(windows_update1_1, vector1_1, theta, window_num)
    time_reddit_size2_fix_adaptive, _ = Fixed_Reference_Clustering_Streaming_Adaptive(windows_update1_2, vector1_2, theta, window_num)
    time_reddit_size3_fix_adaptive, _ = Fixed_Reference_Clustering_Streaming_Adaptive(windows_update1_3, vector1_3, theta, window_num)
    time_reddit_size4_fix_adaptive, _ = Fixed_Reference_Clustering_Streaming_Adaptive(windows_update1_4, vector1_4, theta, window_num)
    time_reddit_size5_fix_adaptive, _ = Fixed_Reference_Clustering_Streaming_Adaptive(windows_update1_5, vector1_5, theta, window_num)

    # Suffice Expire
    time_reddit_size1_Mul_Expire, _ = Multiple_ref_clustering_streaming_Expire(windows_update1_1, vector1_1, theta, window_num, window_size1, para_theta)
    time_reddit_size2_Mul_Expire, _ = Multiple_ref_clustering_streaming_Expire(windows_update1_2, vector1_2, theta, window_num, window_size2, para_theta)
    time_reddit_size3_Mul_Expire, _ = Multiple_ref_clustering_streaming_Expire(windows_update1_3, vector1_3, theta, window_num, window_size3, para_theta)
    time_reddit_size4_Mul_Expire, _ = Multiple_ref_clustering_streaming_Expire(windows_update1_4, vector1_4, theta, window_num, window_size4, para_theta)
    time_reddit_size5_Mul_Expire, _ = Multiple_ref_clustering_streaming_Expire(windows_update1_5, vector1_5, theta, window_num, window_size5, para_theta)

    # Fix Adp的曲线
    average_list_fix_adaptive = [time_reddit_size1_fix_adaptive,
                          time_reddit_size2_fix_adaptive,
                          time_reddit_size3_fix_adaptive,
                          time_reddit_size4_fix_adaptive,
                          time_reddit_size5_fix_adaptive]
    average_window_size_fix_adaptive = []
    for a_list in average_list_fix_adaptive:
        avg = sum(a_list) / len(a_list)
        average_window_size_fix_adaptive.append(avg)

    # Mul Expire的曲线
    average_list_mul_exp = [time_reddit_size1_Mul_Expire, time_reddit_size2_Mul_Expire, time_reddit_size3_Mul_Expire,
                         time_reddit_size4_Mul_Expire,
                         time_reddit_size5_Mul_Expire]
    average_window_size_mul_exp = []
    for a_list in average_list_mul_exp:
        avg = sum(a_list) / len(a_list)
        average_window_size_mul_exp.append(avg)

    print("Fix Adaptive: ", average_window_size_fix_adaptive)
    print("Mul Expire: ", average_window_size_mul_exp)
