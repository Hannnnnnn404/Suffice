import math
import torch
import time
from Clustering import compute_angle


# 静态插入
def Naive_clustering(vectors, theta):
    clusters = []  # 多个vector list组成，每个list代表一个cluster
    angle_degrees = torch.tensor(theta)  # 角度 5°
    threshold = torch.deg2rad(angle_degrees)  # 转换为弧度
    time_each_insert = []

    for i in range(len(vectors)):
        t1 = time.time()
        vector_i = vectors[i]

        # 遍历所有cluster， 看vector适合加入哪个cluster
        min_angle = math.inf
        cluster_to_insert = None
        for cluster in clusters:
            for vector in cluster:  # 和cluster中的每个vector都做比较，看最近的cluster
                angle = compute_angle(vector, vector_i)
                if angle <= min_angle:
                    min_angle = angle
                    # 记录最小的cluster
                    cluster_to_insert = cluster

        # 已知加入哪个cluster
        INSERT = False
        angle_with_all_vector = []
        if cluster_to_insert is not None:
            # 看是否能加如这个cluster
            for vector in cluster_to_insert:
                angle_with_vector = compute_angle(vector, vector_i)
                angle_with_all_vector.append(angle_with_vector)
            if max(angle_with_all_vector) <= threshold:  # 最大夹角小于阈值，插入
                INSERT = True

        if INSERT:  # 能插入
            cluster_to_insert.append(vector_i)
            t2 = time.time()
            print(t2 - t1)
            time_each_insert.append(t2 - t1)
            continue

        # 不能插入，单独开一个新的聚类插入
        new_cluster = [vector_i]
        clusters.append(new_cluster)
        t2 = time.time()
        print(t2 - t1)
        time_each_insert.append(t2 - t1)
    print(len(clusters))
    return time_each_insert


def Naive_clustering_Streaming(windows_updates, vectors, theta, window_num):
    """
    对于每个Streaming vectors，执行完整的Naive插入和删除
    :param windows_updates:
    :return:
    """
    clusters = []  # 多个index list组成，每个list代表一个cluster
    angle_degrees = torch.tensor(theta)  # 角度 5°
    threshold = torch.deg2rad(angle_degrees)  # 转换为弧度
    time_each_window = []
    vector_to_cluster = {}  # 每个vector对应的cluster_id
    time_full_insert = []

    for window_index, updates in windows_updates.items():
        if window_index >= window_num:
            break
        print(f"Processing updates for window {window_index}")
        updated_vectors = {}  # 在这个window变化的向量
        updated_vector_index = set()
        # 记录上一个没更新之前的向量
        vectors_last = vectors

        # updates 是一个 DataFrame，且包含需要的更新信息
        for _, update in updates.iterrows():
            row = update['row_index']  # 需要更新的向量索引
            col = update['col_index']  # 需要更新的维度索引
            behave = update['behave']  # 过期/新增

            # 如果新增
            if behave == 1:
                vectors[row][col] += 1

            elif behave == -1:  # 如果过期，一定在范围内
                vectors[row][col] -= 1

            updated_vector_index.add(row)

        t1 = time.time()
        for row in updated_vector_index:
            updated_vectors[row] = vectors[row]


        if len(clusters) != 0 and len(vectors) != 0:
            for row in updated_vectors.keys():
                if row in vector_to_cluster.keys():
                    clusters[vector_to_cluster[row]].remove(row)

        clusters, vector_to_cluster, time_each_insert = Naive_Clustering_incremental(vectors, updated_vectors, clusters, threshold,
                                                                   vector_to_cluster)
        t2 = time.time()
        print(t2 - t1)
        time_each_window.append(t2 - t1)
        time_full_insert += time_each_insert

    print("几个vector", len(vectors))
    num = 0
    for cluster in clusters:
        if len(cluster) > 0:
            num += 1
            print("几个vector", len(cluster))
    print("几个cluster", num)
    return time_each_window, time_full_insert


def Naive_Clustering_incremental(vectors, updated_vectors, clusters, threshold, vector_in_cluster):
    time_each_insert = []
    for row, vector_i in updated_vectors.items():
        # 遍历所有cluster， 看vector适合加入哪个cluster
        t1 = time.time()
        min_angle = math.inf
        cluster_to_insert = None
        cluster_to_insert_id = -1
        for i in range(len(clusters)):
            for vector_index in clusters[i]:  # 和cluster中的每个vector都做比较，看最近的cluster
                angle = compute_angle(vectors[vector_index], vector_i)
                if angle <= min_angle:
                    min_angle = angle
                    # 记录最小的cluster
                    cluster_to_insert = clusters[i]
                    cluster_to_insert_id = i

        # 已知加入哪个cluster
        INSERT = False
        angle_with_all_vector = []
        if cluster_to_insert is not None:
            # 看是否能加如这个cluster
            for vector_index in cluster_to_insert:
                angle_with_vector = compute_angle(vectors[vector_index], vector_i)
                angle_with_all_vector.append(angle_with_vector)
            if max(angle_with_all_vector) <= threshold:  # 最大夹角小于阈值，插入
                INSERT = True

        if INSERT:  # 能插入
            cluster_to_insert.append(row)
            vector_in_cluster[row] = cluster_to_insert_id
            t2 = time.time()
            print(t2 - t1)
            time_each_insert.append(t2 - t1)
            continue

        # 不能插入，单独开一个新的聚类插入
        new_cluster = [row]
        clusters.append(new_cluster)
        vector_in_cluster[row] = len(clusters) - 1
        t2 = time.time()
        print(t2 - t1)
        time_each_insert.append(t2 - t1)

    return clusters, vector_in_cluster, time_each_insert


def Naive_Clustering_synthetic(vectors, window_updates, theta, window_num):
    clusters = []  # 多个index list组成，每个list代表一个cluster
    angle_degrees = torch.tensor(theta)  # 角度 5°
    threshold = torch.deg2rad(angle_degrees)  # 转换为弧度
    time_each_window = []
    vector_to_cluster = {}  # 每个vector对应的cluster_id
    time_full_insert = []

    # 构造只有value变化的数据集，每个window在不断地更新几个value
    for window_index, updated_vectors in window_updates.items():
        if window_index >= window_num:
            break
        print(f"Processing updates for window {window_index}")
        # updates是字典，记录的当前窗口更新， row: vector

        t1 = time.time()

        # Naive方法将更新的向量先删除
        if len(clusters) != 0 and len(vectors) != 0:
            for row in updated_vectors.keys():
                # 从特定的cluster删除，如果这个vector有cluster
                if row in vector_to_cluster.keys():
                    clusters[vector_to_cluster[row]].remove(row)

        # 再添加
        clusters, vector_to_cluster, time_each_insert = Naive_Clustering_incremental(vectors, updated_vectors, clusters,
                                                                                     threshold,
                                                                                     vector_to_cluster)
        t2 = time.time()
        print(t2 - t1)
        time_each_window.append(t2 - t1)
        time_full_insert += time_each_insert

        # print("几个vector", len(vectors))
        # num = 0
        # for cluster in clusters:
        #     if len(cluster) > 0:
        #         num += 1
        #         print("vector", len(cluster))
        # print("cluster", num)
    return time_each_window, time_full_insert



def Naive_clustering_Streaming_Cluster(windows_updates, vectors, theta, window_num):
    """
    对于每个Streaming vectors，执行完整的Naive插入和删除
    :param windows_updates:
    :return:
    """
    clusters = []  # 多个index list组成，每个list代表一个cluster
    angle_degrees = torch.tensor(theta)  # 角度 5°
    threshold = torch.deg2rad(angle_degrees)  # 转换为弧度
    time_each_window = []
    vector_to_cluster = {}  # 每个vector对应的cluster_id
    time_full_insert = []

    for window_index, updates in windows_updates.items():
        if window_index >= window_num:
            break
        print(f"Processing updates for window {window_index}")
        updated_vectors = {}  # 在这个window变化的向量
        updated_vector_index = set()
        # 记录上一个没更新之前的向量
        vectors_last = vectors

        # updates 是一个 DataFrame，且包含需要的更新信息
        for _, update in updates.iterrows():
            row = update['row_index']  # 需要更新的向量索引
            col = update['col_index']  # 需要更新的维度索引
            behave = update['behave']  # 过期/新增

            # 如果新增
            if behave == 1:
                vectors[row][col] += 1

            elif behave == -1:  # 如果过期，一定在范围内
                vectors[row][col] -= 1

            updated_vector_index.add(row)

        t1 = time.time()
        for row in updated_vector_index:
            updated_vectors[row] = vectors[row]

        # Naive方法将更新的向量先删除
        if len(clusters) != 0 and len(vectors) != 0:
            for row in updated_vectors.keys():
                # 从特定的cluster删除，如果这个vector有cluster
                if row in vector_to_cluster.keys():
                    clusters[vector_to_cluster[row]].remove(row)

        # 再添加
        clusters, vector_to_cluster, time_each_insert = Naive_Clustering_incremental(vectors, updated_vectors, clusters, threshold,
                                                                   vector_to_cluster)
        t2 = time.time()
        print(t2 - t1)
        time_each_window.append(t2 - t1)
        time_full_insert += time_each_insert

    print("vector", len(vectors))
    num = 0
    for cluster in clusters:
        if len(cluster) > 0:
            num += 1
            print("vector", len(cluster))
    print("cluster", num)
    vectors_torch = [torch.tensor(v, dtype=torch.float32) for v in vectors]
    clusters_non_empty = [c for c in clusters if len(c) > 0]
    return vectors_torch, clusters_non_empty


