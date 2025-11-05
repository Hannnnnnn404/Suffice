import copy
import math

import numpy as np
from Clustering import Vector, Reference, Cluster, compute_angle, find_nearest_reference, add_hash_ref2, \
    remove_hash_ref2, check_hash2_bucket, check_hash1_bucket, add_hash_ref1
import torch
import time


def Multiple_ref_clustering_streaming_Global_threshold(windows_updates, vectors, theta, window_num, para_theta, prev_intra, prev_inter, window_K, eta=0.15):
    clusters = []
    angle_degrees = torch.tensor(theta)  # 角度 5°
    threshold = torch.deg2rad(angle_degrees)  # 转换为弧度
    vector_objects = {}  # 对象
    vector_to_cluster = {}
    time_each_window = []
    time_full_insert = []
    hash_table_1_Ref = {i: [] for i in range(10)}
    hash_table_2_Ref = {i: [] for i in range(18)}
    # base_R1 = None
    base_R1 = Reference(torch.randn(vectors[0].shape[0]))
    proj_R = None
    theta_new = theta

    for window_index, updates in windows_updates.items():
        if window_index >= window_num:
            break
        print(f"Processing updates for window {window_index}")
        updated_vectors = {}  # 在这个window变化的向量
        updated_vector_index = set()
        vectors_last = [v.clone() for v in vectors]
        # vectors_last = vectors.copy()

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

        # 更新的vector
        for row in updated_vector_index:
            updated_vectors[row] = vectors[row]

        # 如果只有微小扰动
        to_remove = []
        for row, vector in updated_vectors.items():
            if row in vector_to_cluster.keys():  # vector object还没更新，还是之前窗口的vector object
                updated_angle = compute_angle(vectors_last[row], vector)
                # if torch.equal(vectors_last[row], vector):
                #     print("相等")
                if updated_angle <= torch.deg2rad(torch.tensor(para_theta)):
                    vector_objects[row].data = vector
                    to_remove.append(row)

        # 在循环外修改字典
        for key in to_remove:
            del updated_vectors[key]
            # print("delete")

        # t1 = time.time()
        # 将更新的向量先删除,从cluster中删除
        if len(clusters) != 0 and len(vectors) != 0:
            for row in updated_vectors.keys():
                # 从特定的cluster删除，如果这个vector有cluster
                if row in vector_to_cluster.keys():
                    clusters[vector_to_cluster[row]].delete_vector(vector_objects[row])
                    clusters[vector_to_cluster[row]].remove_hash_table1(vector_objects[row])
                    clusters[vector_to_cluster[row]].remove_hash_table2(vector_objects[row])

        t1 = time.time()
        # 再添加
        vector_objects, clusters, vector_to_cluster, time_each_insert, hash_table_1_Ref, base_R1, hash_table_2_Ref, proj_R\
            = Multiple_ref_clustering_incremental_global_threshold(
            vector_objects,
            updated_vectors,
            clusters,
            threshold, vector_to_cluster, hash_table_1_Ref, base_R1, hash_table_2_Ref, proj_R)
        t2 = time.time()
        print(t2 - t1)
        time_each_window.append(t2 - t1)
        time_full_insert += time_each_insert

        # ==== periodically check θ ====
        if (window_index + 1) % window_K == 0 and len(clusters) > 1:
            theta_new, I_t, E_t, need_recluster = update_global_theta(clusters, theta, prev_intra, prev_inter, eta)
            prev_intra, prev_inter = I_t, E_t
            if need_recluster:
                print(f"[θ changed {theta:.4f} → {theta_new:.4f}] restarting from window 0")
                break  # 退出当前 for 循环，重新开始 while


    print("几个vector", len(vectors))
    num = 0
    for cluster in clusters:
        if len(cluster.vectors) > 0:
            num += 1
            print("几个ref", len(cluster.references), "几个vec", len(cluster.vectors))
    print("几个cluster", num)

    return time_each_window, time_full_insert, prev_intra, prev_inter, theta_new


def Multiple_ref_clustering_incremental_global_threshold(vector_objects, updated_vectors, clusters, threshold, vector_to_cluster,
                                        hash_table_1_Ref, base_R1, hash_table_2_Ref, proj_R):
    time_each_insert = []
    for row, vector_data in updated_vectors.items():
        vector_i = Vector(vector_data)
        vector_objects[row] = vector_i
        # 使用hash search寻找合适的cluster加入
        cluster_to_insert = None
        cluster_to_insert_id = None
        bucket1 = None
        # min_angle_R1 = math.inf

        t1 = time.time()
        # time1 = time.time()
        if len(clusters) != 0 and base_R1 is not None:
            # 检查当前vector最近的cluster
            bucket1 = check_hash1_bucket(hash_table_1_Ref, vector_i, base_R1, 10, threshold)
            near_ref, near_slot, min_angle_R1 = find_nearest_reference(bucket1, hash_table_1_Ref, vector_i)
            cluster_to_insert_id = near_ref[0]
            cluster_to_insert = clusters[cluster_to_insert_id]
        # for i in range(len(clusters)):
        #     angle_with_R1 = compute_angle(vector_i.data, clusters[i].R1.vector.data)
        #     if angle_with_R1 < min_angle_R1:
        #         min_angle_R1 = angle_with_R1
        #         cluster_to_insert = clusters[i]
        #         cluster_to_insert_id = i
        #     break
        # time2 = time.time()
        # print("Search Time", time2 - time1)
        # print(min_angle_R1)
        # if bucket1 != 9:
        # print(bucket1)

        # 已知加入哪个cluster
        INSERT = False
        if cluster_to_insert is not None:
            INSERT, ref_to_insert, insert_angle = cluster_to_insert.check_threshold_to_insert_cluster(vector_i)
        if INSERT:  # 能插入
            # print("YES, insert to existing cluster")
            cluster_to_insert.add_vector(vector_i)
            cluster_to_insert.update_old_references(vector_i)
            max_angle_vector = cluster_to_insert.search_max_angle(vector_i)  # search时间太长
            cluster_to_insert.add_new_reference(vector_i, max_angle_vector, 10)  # 是否能成为新的reference,已经有旧的ref了
            vector_to_cluster[row] = cluster_to_insert_id
        else:  # 不能插入，单独开一个新的聚类插入
            new_cluster = Cluster(threshold)  # 新的Cluster类
            new_cluster.add_vector(vector_i)
            new_cluster.update_old_references(vector_i)
            clusters.append(new_cluster)
            vector_to_cluster[row] = len(clusters) - 1
            # base-R1形成
            if len(clusters) == 1 and base_R1 is None:
                base_R1 = new_cluster.R1
            # # 至少有两个vector，形成proj投影
            # if len(vector_objects) == 2 and proj_R is None:
            #     keys = list(vector_objects.keys())
            #     key2 = keys[1]
            #     val2 = vector_objects[key2]
            #     is_eq = base_R1.vector.data.equal(val2.data)
            #     if is_eq:
            #         print("相等")
            #     # v2在base R1上的投影
            #     proj = (torch.dot(val2.data, base_R1.vector.data) / torch.dot(base_R1.vector.data,base_R1.vector.data)) * base_R1.vector.data
            #     proj_R = proj - val2.data  # 垂直分量

            # 每个R1形成，都加入hash table
            if bucket1 is None:
                bucket1 = 0
            add_hash_ref1(hash_table_1_Ref, new_cluster.R1, bucket1, len(clusters) - 1)



        t2 = time.time()
        time_each_insert.append(t2 - t1)

    return vector_objects, clusters, vector_to_cluster, time_each_insert, hash_table_1_Ref, base_R1, hash_table_2_Ref, proj_R


def update_global_theta(
    clusters, theta, prev_intra, prev_inter, eta=0.15
):
    """
    每 K 个窗口后调用，计算全局簇内 / 簇间相似度，
    根据 intra/inter 的相对变化调整全局 theta，并重新聚类。
    """
    if len(clusters) <= 1:
        return theta, prev_intra, prev_inter, clusters, {}, {}, None

    # === 计算全局簇内 / 簇间相似度 ===
    intra_vals, inter_vals = [], []
    for i, C in enumerate(clusters):
        sims = [torch.dot(v.data, C.R1.vector.data) /
                (torch.norm(v.data) * torch.norm(C.R1.vector.data))
                for v in C.vectors]
        intra_vals.append(float(sum(sims) / len(sims)))

        max_inter = -1.0
        for j, C2 in enumerate(clusters):
            if i == j:
                continue
            sim = float(torch.dot(C.R1.vector.data, C2.R1.vector.data) /
                        (torch.norm(C.R1.vector.data) * torch.norm(C2.R1.vector.data)))
            if sim > max_inter:
                max_inter = sim
        inter_vals.append(max_inter)

    I_t = sum(intra_vals) / len(intra_vals)
    E_t = sum(inter_vals) / len(inter_vals)

    # 判断是否更新 θ
    new_theta = theta
    need_recluster = False

    if prev_intra is not None and prev_inter is not None:
        u = I_t / prev_intra - 1.0
        v = E_t / prev_inter - 1.0

        # 按你最新规则
        if u > 0 and v > 0:  # intra↑, inter↑ → 放宽
            c = v
        elif u > 0 and v <= 0:  # intra↑, inter↓ → 不更新
            c = 0.0
        elif u <= 0:  # intra↓ → 收紧
            c = u
        else:
            c = 0.0

        if c != 0.0:
            new_theta = theta * (1.0 + eta * c)
            need_recluster = True
            print(f"[θ update triggered] I={I_t:.4f}, E={E_t:.4f}, u={u:.4f}, v={v:.4f}, c={c:.4f}, θ→{new_theta:.4f}")
        else:
            print(f"[θ stable] I={I_t:.4f}, E={E_t:.4f}, no update")

    else:
        print(f"[θ init] I={I_t:.4f}, E={E_t:.4f} (first baseline set)")

    return new_theta, I_t, E_t, need_recluster

