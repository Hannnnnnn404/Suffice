import torch
import numpy as np
from math import degrees
from dataset.Slinding_window import process_streaming_data

from Links import LinksStream_Cluster
from HSRC import HSRCStream_Cluster
from D_Stream import DStream_Cluster
from DEN_Stream import DENStream_Cluster
from Naive_Clustering import Naive_clustering_Streaming_Cluster
from Fixed_Reference_Clustering import Fixed_Reference_Clustering_Streaming_Cluster
from Multiple_Reference_Clustering import Multiple_ref_clustering_streaming_Cluster
import torch
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import math
from Fix_Dynamic import Fixed_Reference_Clustering_Streaming_Adaptive_Cluster
from Multiple_Reference_Clustering_Expire import Multiple_ref_clustering_streaming_Expire_Cluster

def cosine_based_silhouette(vectors: list[torch.Tensor], cluster_assignments: list[list[int]]) -> float:
    labels = np.zeros(len(vectors), dtype=int)
    for cid, cluster in enumerate(cluster_assignments):
        for idx in cluster:
            labels[idx] = cid

    vectors_np = np.stack([v.numpy() for v in vectors])
    return silhouette_score(vectors_np, labels, metric="cosine")


def average_intra_cluster_similarity(vectors, clusters):
    X = torch.stack(vectors)
    total_sim = 0.0
    count = 0
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                u, v = X[cluster[i]], X[cluster[j]]
                total_sim += torch.dot(u, v) / (torch.norm(u) * torch.norm(v))
                count += 1
    return total_sim.item() / count if count > 0 else 0.0


def average_inter_cluster_similarity(vectors, clusters):
    X = torch.stack(vectors)
    total_sim = 0.0
    count = 0
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            for u_id in clusters[i]:
                for v_id in clusters[j]:
                    u, v = X[u_id], X[v_id]
                    total_sim += torch.dot(u, v) / (torch.norm(u) * torch.norm(v))
                    count += 1
    return total_sim.item() / count if count > 0 else 0.0


def max_angle_in_cluster(vectors, clusters):
    X = torch.stack(vectors)
    max_angle = 0.0
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                u, v = X[cluster[i]], X[cluster[j]]
                cosine = torch.clamp(torch.dot(u, v) / (torch.norm(u) * torch.norm(v)), -1.0, 1.0)
                angle = torch.acos(cosine).item()
                if angle > max_angle:
                    max_angle = angle
    return math.degrees(max_angle)


if __name__ == '__main__':
    updates_path1 = 'dataset/BGP'
    window_size1 = 500
    # 不同数据集的疏通window size
    windows_update1_1, vectors1 = process_streaming_data(updates_path1, window_size1)
    theta = 30.0
    window_num = 1000

    vectors_reddit_links, cluster_reddit_size1_links = LinksStream_Cluster(windows_update1_1, vectors1, Tc=1.0,
                                                                           Ts=1.0, window_num=window_num)
    results_links = {
        "cosine_silhouette": cosine_based_silhouette(vectors_reddit_links, cluster_reddit_size1_links),
        "intra_sim": average_intra_cluster_similarity(vectors_reddit_links, cluster_reddit_size1_links),
        "inter_sim": average_inter_cluster_similarity(vectors_reddit_links, cluster_reddit_size1_links),
        "max_angle": max_angle_in_cluster(vectors_reddit_links, cluster_reddit_size1_links)
    }
    print("Links: ", results_links)

    vectors_reddit_hsrc, cluster_reddit_size1_hsrc = HSRCStream_Cluster(windows_updates=windows_update1_1,
                                                                        vectors=vectors1,
                                                                        theta=theta,  # cosine 角度阈值
                                                                        k=10,  # KNN邻居数
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
    print("HSRC: ", results_hsrc)


    vectors_reddit_dstream, cluster_reddit_size1_dstream = DStream_Cluster(windows_update1_1, vectors1, theta,
                                                                           window_num)
    results_dstream = {
        "cosine_silhouette": cosine_based_silhouette(vectors_reddit_dstream, cluster_reddit_size1_dstream),
        "intra_sim": average_intra_cluster_similarity(vectors_reddit_dstream, cluster_reddit_size1_dstream),
        "inter_sim": average_inter_cluster_similarity(vectors_reddit_dstream, cluster_reddit_size1_dstream),
        "max_angle": max_angle_in_cluster(vectors_reddit_dstream, cluster_reddit_size1_dstream)
    }
    print(results_dstream)


    vectors_reddit_denstream, cluster_reddit_size1_denstream = DENStream_Cluster(windows_update1_1, vectors1, theta,
                                                                                 window_num)
    results_denstream = {
        "cosine_silhouette": cosine_based_silhouette(vectors_reddit_denstream, cluster_reddit_size1_denstream),
        "intra_sim": average_intra_cluster_similarity(vectors_reddit_denstream, cluster_reddit_size1_denstream),
        "inter_sim": average_inter_cluster_similarity(vectors_reddit_denstream, cluster_reddit_size1_denstream),
        "max_angle": max_angle_in_cluster(vectors_reddit_denstream, cluster_reddit_size1_denstream)
    }
    print("DenStream: ", results_denstream)


    vectors_reddit_naive, cluster_reddit_size1_naive = Naive_clustering_Streaming_Cluster(windows_update1_1, vectors1,
                                                                                          theta, window_num)
    results_naive = {
        "cosine_silhouette": cosine_based_silhouette(vectors_reddit_naive, cluster_reddit_size1_naive),
        "intra_sim": average_intra_cluster_similarity(vectors_reddit_naive, cluster_reddit_size1_naive),
        "inter_sim": average_inter_cluster_similarity(vectors_reddit_naive, cluster_reddit_size1_naive),
        "max_angle": max_angle_in_cluster(vectors_reddit_naive, cluster_reddit_size1_naive)
    }
    print("Naive: ", results_naive)


    vectors_reddit_fix, cluster_reddit_size1_fix = Fixed_Reference_Clustering_Streaming_Cluster(windows_update1_1,
                                                                                                vectors1, theta,
                                                                                                window_num)
    results_fix = {
        "cosine_silhouette": cosine_based_silhouette(vectors_reddit_fix, cluster_reddit_size1_fix),
        "intra_sim": average_intra_cluster_similarity(vectors_reddit_fix, cluster_reddit_size1_fix),
        "inter_sim": average_inter_cluster_similarity(vectors_reddit_fix, cluster_reddit_size1_fix),
        "max_angle": max_angle_in_cluster(vectors_reddit_fix, cluster_reddit_size1_fix)
    }
    print("Fix: ", results_fix)

    vectors_reddit_mul, cluster_reddit_size1_mul = Multiple_ref_clustering_streaming_Cluster(windows_update1_1, vectors1, theta, window_num, 10.0)
    results_mul = {
        "cosine_silhouette": cosine_based_silhouette(vectors_reddit_mul, cluster_reddit_size1_mul),
        "intra_sim": average_intra_cluster_similarity(vectors_reddit_mul, cluster_reddit_size1_mul),
        "inter_sim": average_inter_cluster_similarity(vectors_reddit_mul, cluster_reddit_size1_mul),
        "max_angle": max_angle_in_cluster(vectors_reddit_mul, cluster_reddit_size1_mul)
    }

    print("Links: ", results_links)
    print("HSRC: ", results_hsrc)
    print("DenStream: ", results_denstream)
    print("Naive: ", results_naive)
    print("Fix: ", results_fix)
    print("mul: ", results_mul)

    vectors_reddit_fix_adaptive, cluster_reddit_size1_fix_adaptive = Fixed_Reference_Clustering_Streaming_Adaptive_Cluster(windows_update1_1,
                                                                                                vectors1, theta,
                                                                                                window_num)
    results_fix_adaptive = {
        "cosine_silhouette": cosine_based_silhouette(vectors_reddit_fix_adaptive, cluster_reddit_size1_fix_adaptive),
        "intra_sim": average_intra_cluster_similarity(vectors_reddit_fix_adaptive, cluster_reddit_size1_fix_adaptive),
        "inter_sim": average_inter_cluster_similarity(vectors_reddit_fix_adaptive, cluster_reddit_size1_fix_adaptive),
        "max_angle": max_angle_in_cluster(vectors_reddit_fix_adaptive, cluster_reddit_size1_fix_adaptive)
    }
    print("Fix Adaptive: ", results_fix_adaptive)

    vectors_reddit_mul_exp, cluster_reddit_size1_mul_exp = Multiple_ref_clustering_streaming_Expire_Cluster(windows_update1_1,
                                                                                             vectors1, theta,
                                                                                             window_num,  window_size1, 10.0)
    results_mul_exp = {
        "cosine_silhouette": cosine_based_silhouette(vectors_reddit_mul_exp, cluster_reddit_size1_mul_exp),
        "intra_sim": average_intra_cluster_similarity(vectors_reddit_mul_exp, cluster_reddit_size1_mul_exp),
        "inter_sim": average_inter_cluster_similarity(vectors_reddit_mul_exp, cluster_reddit_size1_mul_exp),
        "max_angle": max_angle_in_cluster(vectors_reddit_mul_exp, cluster_reddit_size1_mul_exp)
    }
    print("Fix Adaptive: ", results_fix_adaptive)
    print("mul Expire: ", results_mul_exp)






