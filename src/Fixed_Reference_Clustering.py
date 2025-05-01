import math
import torch
import numpy as np
import time
from Clustering import compute_angle
from collections import defaultdict


def Fixed_ref_clustering(vectors, theta):
    c_id = 0
    clusters = defaultdict(list)  # 多个vector list组成，每个list代表一个cluster
    references = {}  # cluster_id: reference
    angle_degrees = torch.tensor(theta)  # 角度 5°
    threshold = torch.deg2rad(angle_degrees)  # 转换为弧度
    time_each_insert = []

    for i in range(len(vectors)):
        t1 = time.time()
        vector_i = vectors[i]

        # 遍历所有cluster， 看vector适合加入哪个cluster
        min_angle = math.inf
        cluster_to_insert = None
        for c_id, cluster in clusters.items():
            ref = references[c_id]
            angle = compute_angle(ref, vector_i)
            if angle <= min_angle:
                min_angle = angle
                cluster_to_insert = c_id

        # 已知加入哪个cluster
        INSERT = False
        Statisfy_Cluster = False
        if cluster_to_insert is not None:
            # 看是否能加如这个cluster
            angle = compute_angle(references[cluster_to_insert], vector_i)
            if angle <= threshold / 2:  # 最大夹角小于阈值，插入
                INSERT = True

        if INSERT:  # 能插入,下一个
            clusters[cluster_to_insert].append(vector_i)
            t2 = time.time()
            time_each_insert.append(t2 - t1)
            continue
        elif cluster_to_insert is not None:
            # 不能插入，大于ref阈值，但仍满足聚类插入
            max_angle_with_vector = 0
            for v in clusters[cluster_to_insert]:
                angle = compute_angle(vector_i, v)
                if angle > max_angle_with_vector:
                    max_angle_with_vector = angle
            if max_angle_with_vector < threshold:
                Statisfy_Cluster = True

        if Statisfy_Cluster:  # 能插入聚类
            clusters[cluster_to_insert].append(vector_i)
            t2 = time.time()
            print(t2 - t1)
            time_each_insert.append(t2 - t1)
        else:
            # 不能插入，单独开一个新的聚类插入
            c_id += 1
            clusters[c_id].append(vector_i)
            references[c_id] = vector_i  # 新聚类的第一个作为reference
            t2 = time.time()
            print(t2 - t1)
            time_each_insert.append(t2 - t1)
    print("Fixed方法", len(clusters))
    return time_each_insert


def Fixed_Reference_Clustering_Streaming(windows_updates, vectors, theta, window_num):
    clusters = defaultdict(list)  # 多个vector index list组成，每个list代表一个cluster
    references = {}  # cluster_id: reference
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

        # t1 = time.time()
        # Naive方法将更新的向量先删除
        if len(clusters) != 0 and len(vectors) != 0:
            for row in updated_vectors.keys():
                # 从特定的cluster删除，如果这个vector有cluster
                if row in vector_to_cluster.keys():
                    clusters[vector_to_cluster[row]].remove(row)

        # 再添加
        references, clusters, vector_to_cluster, time_each_insert = Fixed_Reference_Clustering_Incremental(vectors,
                                                                                                           updated_vectors,
                                                                                                           references,
                                                                                                           clusters,
                                                                                                           threshold,
                                                                                                           vector_to_cluster)
        t2 = time.time()
        print(t2 - t1)
        time_each_window.append(t2 - t1)
        time_full_insert += time_each_insert
    return time_each_window, time_full_insert


def Fixed_Reference_Clustering_Incremental(vectors, updated_vectors, references, clusters, threshold,
                                           vector_in_cluster):
    c_id = 0
    time_each_insert = []
    # clusters是一个字典，key是c_id, value是一个list，里面存储vector的索引
    for row, vector_i in updated_vectors.items():
        t1 = time.time()
        # 遍历所有cluster， 看vector适合加入哪个cluster
        min_angle = math.inf
        cluster_to_insert = None
        time1 = time.time()
        print(len(clusters.keys()))
        for c_id, cluster in clusters.items():
            ref = references[c_id]
            angle = compute_angle(ref, vector_i)
            if angle <= min_angle:
                min_angle = angle
                cluster_to_insert = c_id
        time2 = time.time()
        print("Search cluster time", time2 - time1)

        # 已知加入哪个cluster
        INSERT = False
        Statisfy_Cluster = False
        if cluster_to_insert is not None:
            # 看是否能加如这个cluster
            # 先计算reference的阈值，也就是theta - max angle
            # 计算max angle使用Naive方法
            # max_angle_vector = 0
            # for v in clusters[cluster_to_insert]:
            #     angle_between_vector = compute_angle(vector_i, vectors[v])
            #     if angle_between_vector > max_angle_vector:
            #         max_angle_vector = angle_between_vector
            # angle = compute_angle(references[cluster_to_insert], vector_i)
            # if angle <= threshold / 2:  # 最大夹角小于阈值，插入
            #     INSERT = True

            # 原写法
            angle = compute_angle(references[cluster_to_insert], vector_i)
            if angle <= threshold / 2:  # 最大夹角小于阈值，插入
                INSERT = True

        if INSERT:  # 能插入,下一个
            clusters[cluster_to_insert].append(row)
            vector_in_cluster[row] = cluster_to_insert
            t2 = time.time()
            time_each_insert.append(t2 - t1)
            print("Yes,i t insert fix ref")
            print(t2 - t1)
            continue
        elif cluster_to_insert is not None:
            # 不能插入，大于ref阈值，但仍满足聚类插入
            max_angle_with_vector = 0
            for v in clusters[cluster_to_insert]:
                angle = compute_angle(vector_i, vectors[v])
                if angle > max_angle_with_vector:
                    max_angle_with_vector = angle
            if max_angle_with_vector < threshold:
                Statisfy_Cluster = True
                print("Cannot insert by ref, need to search the cluster")

        if Statisfy_Cluster:  # 能插入聚类
            clusters[cluster_to_insert].append(row)
            vector_in_cluster[row] = cluster_to_insert
            t2 = time.time()
            time_each_insert.append(t2 - t1)
            print(t2 - t1)
        else:
            # 不能插入，单独开一个新的聚类插入
            print("Cannot insert by ref, search also fail, need new cluster")
            c_id = len(clusters)
            clusters[c_id].append(row)
            vector_in_cluster[row] = c_id
            references[c_id] = vector_i  # 新聚类的第一个作为reference
            t2 = time.time()
            time_each_insert.append(t2 - t1)
            print(t2 - t1)

    return references, clusters, vector_in_cluster, time_each_insert