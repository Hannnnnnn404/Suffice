import itertools
import random
import time
from dataset.synthetic.Generate_synthetic_data import generate_synthetic_dimension
from Multiple_Reference_Clustering import Multiple_ref_clustering_streaming
from Naive_Clustering import Naive_clustering_Streaming
from Fixed_Reference_Clustering import Fixed_Reference_Clustering_Streaming
from dataset.Slinding_window import process_streaming_syn_data
from D_Stream import DStream
from DEN_Stream import DENStream
from Links import LinksStream
from Fix_Dynamic import Fixed_Reference_Clustering_Streaming_Adaptive
from Multiple_Reference_Clustering_Expire import Multiple_ref_clustering_streaming_Expire
import matplotlib

if __name__ == '__main__':
    theta = 30.0
    window_num = 1000
    window_size = 500
    time_dimension_naive_avg_list = []
    time_dimension_fixed_avg_list = []
    time_dimension_mul_avg_list = []
    time_dimension_dstream_avg_list = []
    time_dimension_denstream_avg_list = []
    time_dimension_links_avg_list = []
    time_dimension_fix_adp_avg_list = []
    time_dimension_mul_exp_avg_list = []
    
    # num_dimensions = [5, 10, 15, 20, 25]
    num_dimensions = [1, 5, 10, 20, 35]
    # for num_dimension in num_dimensions:
    #     generate_synthetic_dimension(num_dimension)
    for num_dim in num_dimensions:
        dimension_root = f"./dataset/synthetic/initial_dimensions_dimension_{num_dim}.pt"
        update_root = f"./dataset/synthetic/generated_dimension_updates_{num_dim}.csv"

        window_updates, dimension = process_streaming_syn_data(dimension_root, update_root, window_size)

        # 方法一： Naive
        time_window_dimension_naive, _ = Naive_clustering_Streaming(window_updates, dimension, theta, window_num)

        # 方法二：Fixed
        time_window_dimension_fix, _ = Fixed_Reference_Clustering_Streaming(window_updates, dimension, theta, window_num)

        # 方法三：mul
        time_window_dimension_mul, _ = Multiple_ref_clustering_streaming(window_updates, dimension, theta, window_num, 0.001)

        time_dimension_naive_avg = sum(time_window_dimension_naive)
        time_dimension_fix_avg = sum(time_window_dimension_fix)
        time_dimension_mul_avg = sum(time_window_dimension_mul)

        time_dimension_naive_avg_list.append(time_dimension_naive_avg)
        time_dimension_fixed_avg_list.append(time_dimension_fix_avg)
        time_dimension_mul_avg_list.append(time_dimension_mul_avg)

        # D-stream
        time_window_dimension_dstream, initial_time = DStream(window_updates, dimension, theta, window_num)

        # DenStream
        time_window_dimension_denstream, _ = DENStream(window_updates, dimension, theta, window_num)

        # Links
        time_window_dimension_links, _ = LinksStream(window_updates, dimension, Tc=0.95, Ts=0.90, window_num=window_num)

        # HSRC

        # Fix Adaptive
        time_window_dimension_fix_adp, _ = Fixed_Reference_Clustering_Streaming_Adaptive(window_updates, dimension, theta,
                                                                                      window_num)

        # Mul-Expire
        time_window_dimension_mul_exp, _ = Multiple_ref_clustering_streaming_Expire(window_updates, dimension, theta,
                                                                                 window_num, window_size, 5.0)

        time_dimension_dstream_avg = sum(time_window_dimension_dstream)
        time_dimension_denstream_avg = sum(time_window_dimension_denstream)
        time_dimension_links_avg = sum(time_window_dimension_links)
        time_dimension_fix_adp_avg = sum(time_window_dimension_fix_adp)
        time_dimension_mul_exp_avg = sum(time_window_dimension_mul_exp)

        time_dimension_dstream_avg_list.append(time_dimension_dstream_avg)
        time_dimension_denstream_avg_list.append(time_dimension_denstream_avg)
        time_dimension_links_avg_list.append(time_dimension_links_avg)
        time_dimension_fix_adp_avg_list.append(time_dimension_fix_adp_avg)
        time_dimension_mul_exp_avg_list.append(time_dimension_mul_exp_avg)

    # 保存数据
    print(time_dimension_naive_avg_list)
    print(time_dimension_fixed_avg_list)
    print(time_dimension_mul_avg_list)
    print("Dstream syn dimension: ", time_dimension_dstream_avg_list)
    print("Denstream syn dimension: ", time_dimension_denstream_avg_list)
    print("Links syn dimension: ", time_dimension_links_avg_list)
    print("Fix-Adp syn dimension: ", time_dimension_denstream_avg_list)
    print("Mil-Exp syn dimension: ", time_dimension_links_avg_list)

   