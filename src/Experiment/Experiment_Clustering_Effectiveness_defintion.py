import itertools
import random
import time
from D_Stream import DStream
from DEN_Stream import DENStream
from Links import LinksStream
from Multiple_Reference_Clustering import Multiple_ref_clustering_streaming
from Naive_Clustering import Naive_clustering_Streaming
from Fixed_Reference_Clustering import Fixed_Reference_Clustering_Streaming
from dataset.Slinding_window import process_streaming_data
from Fix_Dynamic import Fixed_Reference_Clustering_Streaming_Adaptive
from Multiple_Reference_Clustering_Expire import Multiple_ref_clustering_streaming_Expire
from Links_Original import LinksStream_Original_Cluster
from HSRC_Original import HSRCStream_Original_Cluster
from D_Stream_Original import DStream_Original_Cluster
from DEN_Stream_Original import DENStream_Original_Cluster
from Naive_Clustering import Naive_clustering_Streaming_Cluster
from Fixed_Reference_Clustering import Fixed_Reference_Clustering_Streaming_Cluster
from Multiple_Reference_Clustering import Multiple_ref_clustering_streaming_Cluster
from Fix_Dynamic import Fixed_Reference_Clustering_Streaming_Adaptive_Cluster
from Multiple_Reference_Clustering_Expire import Multiple_ref_clustering_streaming_Expire_Cluster
from Experiment_Clsutering_Effectiveness import cosine_based_silhouette, average_inter_cluster_similarity, average_intra_cluster_similarity, max_angle_in_cluster

if __name__ == '__main__':
    updates_path1 = 'dataset/Reddit'
    window_size1 = 500
    # 不同数据集的疏通window size
    windows_update1_1, vectors1 = process_streaming_data(updates_path1, window_size1)
    theta = 30.0
    window_num = 1000

    vectors_reddit_links, cluster_reddit_size1_links = LinksStream_Original_Cluster(windows_update1_1, vectors1, Tc=1.0,
                                                                           Ts=1.0, window_num=window_num)
    results_links = {
        "cosine_silhouette": cosine_based_silhouette(vectors_reddit_links, cluster_reddit_size1_links),
        "intra_sim": average_intra_cluster_similarity(vectors_reddit_links, cluster_reddit_size1_links),
        "inter_sim": average_inter_cluster_similarity(vectors_reddit_links, cluster_reddit_size1_links),
        "max_angle": max_angle_in_cluster(vectors_reddit_links, cluster_reddit_size1_links)
    }
    print("Links: ", results_links)

    vectors_reddit_hsrc, cluster_reddit_size1_hsrc = HSRCStream_Original_Cluster(windows_updates=windows_update1_1,
                                                                        vectors=vectors1,
                                                                        theta=theta,  # cosine 角度阈值
                                                                        s=0.2,  # 代表点比例
                                                                        window_num=window_num,
                                                                        dim=len(vectors1),  # 初始维度
                                                                        device='cpu'  # 或 'cpu'
                                                                        )
    results_hsrc = {
        "cosine_silhouette": cosine_based_silhouette(vectors_reddit_hsrc, cluster_reddit_size1_hsrc),
        "intra_sim": average_intra_cluster_similarity(vectors_reddit_hsrc, cluster_reddit_size1_hsrc),
        "inter_sim": average_inter_cluster_similarity(vectors_reddit_hsrc, cluster_reddit_size1_hsrc),
        "max_angle": max_angle_in_cluster(vectors_reddit_hsrc, cluster_reddit_size1_hsrc)
    }


    vectors_reddit_dstream, cluster_reddit_size1_dstream = DStream_Original_Cluster(windows_update1_1, vectors1,
                                                                           window_num)
    results_dstream = {
        "cosine_silhouette": cosine_based_silhouette(vectors_reddit_dstream, cluster_reddit_size1_dstream),
        "intra_sim": average_intra_cluster_similarity(vectors_reddit_dstream, cluster_reddit_size1_dstream),
        "inter_sim": average_inter_cluster_similarity(vectors_reddit_dstream, cluster_reddit_size1_dstream),
        "max_angle": max_angle_in_cluster(vectors_reddit_dstream, cluster_reddit_size1_dstream)
    }


    vectors_reddit_denstream, cluster_reddit_size1_denstream = DENStream_Original_Cluster(windows_update1_1, vectors1,
                                                                                 window_num)
    results_denstream = {
        "cosine_silhouette": cosine_based_silhouette(vectors_reddit_denstream, cluster_reddit_size1_denstream),
        "intra_sim": average_intra_cluster_similarity(vectors_reddit_denstream, cluster_reddit_size1_denstream),
        "inter_sim": average_inter_cluster_similarity(vectors_reddit_denstream, cluster_reddit_size1_denstream),
        "max_angle": max_angle_in_cluster(vectors_reddit_denstream, cluster_reddit_size1_denstream)
    }

    print("DStream: ",results_dstream)
    print("Links: ", results_links)
    print("HSRC: ", results_hsrc)
    print("DenStream: ", results_denstream)