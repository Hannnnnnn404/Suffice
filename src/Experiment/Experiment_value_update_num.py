
import itertools
import random
import time
from dataset.synthetic.Generate_synthetic_data import generate_synthetic_values
from Multiple_Reference_Clustering import Multiple_ref_clustering_streaming
from Naive_Clustering import Naive_clustering_Streaming
from Fixed_Reference_Clustering import Fixed_Reference_Clustering_Streaming
from dataset.Slinding_window import process_streaming_syn_data
from D_Stream import DStream
from DEN_Stream import DENStream
from Links import LinksStream
from Fix_Dynamic import Fixed_Reference_Clustering_Streaming_Adaptive
from Multiple_Reference_Clustering_Expire import Multiple_ref_clustering_streaming_Expire

if __name__ == '__main__':
    theta = 30.0
    window_num = 1000
    window_size = 500
    # num_values = [25, 50, 75, 100, 125]
    # num_values = [5, 20, 50, 100, 150]
    num_values = [10, 100, 1000, 5000, 10000]
    time_value_naive_avg_list = []
    time_value_fixed_avg_list = []
    time_value_mul_avg_list = []
    time_value_dstream_avg_list = []
    time_value_denstream_avg_list = []
    time_value_links_avg_list = []
    time_value_fix_adp_avg_list = []
    time_value_mul_exp_avg_list = []

    for num_value in num_values:
        vector_root = f"./dataset/synthetic/initial_vectors_value_{num_value}.pt"
        update_root = f"./dataset/synthetic/generated_value_updates_{num_value}.csv"

        window_updates, vector = process_streaming_syn_data(vector_root, update_root, window_size,)

        # 方法一： Naive
        time_window_value_naive, _ = Naive_clustering_Streaming(window_updates, vector, theta, window_num)

        # 方法二：Fixed
        time_window_value_fix, _ = Fixed_Reference_Clustering_Streaming(window_updates, vector, theta, window_num)

        # 方法三：mul
        time_window_value_mul, _ = Multiple_ref_clustering_streaming(window_updates, vector, theta, window_num, 0.001)


        time_value_naive_avg = sum(time_window_value_naive)
        time_value_fix_avg = sum(time_window_value_fix)
        time_value_mul_avg = sum(time_window_value_mul)

        time_value_naive_avg_list.append(time_value_naive_avg)
        time_value_fixed_avg_list.append(time_value_fix_avg)
        time_value_mul_avg_list.append(time_value_mul_avg)

        # D-stream
        time_window_value_dstream, initial_time = DStream(window_updates, vector, theta, window_num)

        # DenStream
        time_window_value_denstream, _ = DENStream(window_updates, vector, theta, window_num)

        # Links
        time_window_value_links, _ = LinksStream(window_updates, vector, Tc=1.0, Ts=1.0, window_num=window_num)

        # HSRC

        # Fix Adaptive
        time_window_value_fix_adp, _ = Fixed_Reference_Clustering_Streaming_Adaptive(window_updates, vector, theta, window_num)

        # Mul-Expire
        time_window_value_mul_exp, _ = Multiple_ref_clustering_streaming_Expire(window_updates, vector, theta, window_num, window_size, 0.001)

        time_value_dstream_avg = sum(time_window_value_dstream)
        time_value_denstream_avg = sum(time_window_value_denstream)
        time_value_links_avg = sum(time_window_value_links)
        time_value_fix_adp_avg = sum(time_window_value_fix_adp)
        time_value_mul_exp_avg = sum(time_window_value_mul_exp)

        time_value_dstream_avg_list.append(time_value_dstream_avg)
        time_value_denstream_avg_list.append(time_value_denstream_avg)
        time_value_links_avg_list.append(time_value_links_avg)
        time_value_fix_adp_avg_list.append(time_value_fix_adp_avg)
        time_value_mul_exp_avg_list.append(time_value_mul_exp_avg)

    # 保存数据
    print("Dstream syn Value: ", time_value_dstream_avg_list)
    print("Denstream syn Value: ", time_value_denstream_avg_list)
    print("Links syn Value: ", time_value_links_avg_list)
    print("Fix-Adp syn Value: ", time_value_denstream_avg_list)
    print("Mil-Exp syn Value: ", time_value_links_avg_list)

