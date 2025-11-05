import itertools
import random
import time
from dataset.synthetic.Generate_synthetic_data import generate_synthetic_vector
from DEN_Stream import DENStream
from D_Stream import DStream
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

matplotlib.use('Agg')

if __name__ == '__main__':
    theta = 30.0
    window_num = 1000
    window_size = 500
    # num_vectors = [25, 50, 75, 100, 125]
    num_vectors = [10, 100, 1000, 5000, 10000]
    time_vector_naive_avg_list = []
    time_vector_fixed_avg_list = []
    time_vector_mul_avg_list = []
    time_vector_dstream_avg_list = []
    time_vector_denstream_avg_list = []
    time_vector_links_avg_list = []
    time_vector_fix_adp_avg_list = []
    time_vector_mul_exp_avg_list = []
    
    # for num_vector in num_vectors:
    #     generate_synthetic_vectors(num_vector)
    for num_vector in num_vectors:
        vector_root = f"./dataset/synthetic/initial_vectors_vector_{num_vector}.pt"
        update_root = f"./dataset/synthetic/generated_vector_updates_{num_vector}.csv"

        window_updates, vector = process_streaming_syn_data(vector_root, update_root, window_size)

        time_window_vetor_dstream,_ = DStream(window_updates, vector, theta, window_num)

        # 方法一： Naive
        time_window_vector_naive, _ = Naive_clustering_Streaming(window_updates, vector, theta, window_num)

        # 方法二：Fixed
        time_window_vector_fix, _ = Fixed_Reference_Clustering_Streaming(window_updates, vector, theta, window_num)

        # 方法三：mul
        time_window_vector_mul, _ = Multiple_ref_clustering_streaming(window_updates, vector, theta, window_num, 5.0)

        time_vector_naive_avg = sum(time_window_vector_naive)
        time_vector_fix_avg = sum(time_window_vector_fix)
        time_vector_mul_avg = sum(time_window_vector_mul)

        time_vector_naive_avg_list.append(time_vector_naive_avg)
        time_vector_fixed_avg_list.append(time_vector_fix_avg)
        time_vector_mul_avg_list.append(time_vector_mul_avg)
        # D-stream
        time_window_vector_dstream, initial_time = DStream(window_updates, vector, theta, window_num)

        # DenStream
        time_window_vector_denstream, _ = DENStream(window_updates, vector, theta, window_num)

        # Links
        time_window_vector_links, _ = LinksStream(window_updates, vector, Tc=1.0, Ts=1.0, window_num=window_num)

        # HSRC

        # Fix Adaptive
        time_window_vector_fix_adp, _ = Fixed_Reference_Clustering_Streaming_Adaptive(window_updates, vector, theta,
                                                                                     window_num)

        # Mul-Expire
        time_window_vector_mul_exp, _ = Multiple_ref_clustering_streaming_Expire(window_updates, vector, theta,
                                                                                window_num, window_size, 5.0)

        time_vector_dstream_avg = sum(time_window_vector_dstream)
        time_vector_denstream_avg = sum(time_window_vector_denstream)
        time_vector_links_avg = sum(time_window_vector_links)
        time_vector_fix_adp_avg = sum(time_window_vector_fix_adp)
        time_vector_mul_exp_avg = sum(time_window_vector_mul_exp)

        time_vector_dstream_avg_list.append(time_vector_dstream_avg)
        time_vector_denstream_avg_list.append(time_vector_denstream_avg)
        time_vector_links_avg_list.append(time_vector_links_avg)
        time_vector_fix_adp_avg_list.append(time_vector_fix_adp_avg)
        time_vector_mul_exp_avg_list.append(time_vector_mul_exp_avg)

    print("Dstream syn vector: ", time_vector_dstream_avg_list)
    print("Denstream syn vector: ", time_vector_denstream_avg_list)
    print("Links syn vector: ", time_vector_links_avg_list)
    print("Fix-Adp syn vector: ", time_vector_denstream_avg_list)
    print("Mil-Exp syn vector: ", time_vector_links_avg_list)



