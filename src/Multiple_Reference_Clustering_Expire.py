import copy
import math

import numpy as np
from Clustering import Vector, Reference, Cluster, compute_angle, find_nearest_reference, add_hash_ref2, \
    remove_hash_ref2, check_hash2_bucket, check_hash1_bucket, add_hash_ref1
import torch
import time


def Multiple_ref_clustering_streaming_Expire(windows_updates, vectors, theta, window_num, window_size, para_theta):
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
    ref_status = {}

    for window_index, updates in windows_updates.items():
        if window_index >= window_num:
            break
        print(f"Processing updates for window {window_index}")
        updated_vectors = {}  # 在这个window变化的向量
        updated_vector_index = set()
        # 记录上一个没更新之前的向量
        vectors_last = [v.clone() for v in vectors]
        # vectors_last = vectors.copy()

        for _, update in updates.iterrows():
            row = update['row_index']
            col = update['col_index']
            behave = update['behave']

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
            if row in vector_to_cluster.keys():
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
        vector_objects, clusters, vector_to_cluster, time_each_insert, hash_table_1_Ref, base_R1, hash_table_2_Ref, proj_R, ref_status \
            = Multiple_ref_clustering_incremental_expire(
            vector_objects, updated_vectors, clusters, threshold, vector_to_cluster,
            hash_table_1_Ref, base_R1, hash_table_2_Ref, proj_R,
            window_index, ref_status)
        # === Step 6️: expire references after each window ===
        ref_status = expire_references(ref_status, window_index, window_size)
        t2 = time.time()
        print(t2 - t1)
        time_each_window.append(t2 - t1)
        time_full_insert += time_each_insert

    print("几个vector", len(vectors))
    num = 0
    for cluster in clusters:
        if len(cluster.vectors) > 0:
            num += 1
            print("几个ref", len(cluster.references), "几个vec", len(cluster.vectors))
    print("几个cluster", num)

    return time_each_window, time_full_insert


def Multiple_ref_clustering_incremental_expire(
        vector_objects, updated_vectors, clusters, threshold, vector_to_cluster,
        hash_table_1_Ref, base_R1, hash_table_2_Ref, proj_R,
        current_window, ref_status):
    """
    Incremental multi-reference clustering with external reference expiration (Suffice-Expire)
    - References managed by an external dictionary ref_status[(cluster_id, ref_id)]
    - Expiration rule: reference expires if not used for an entire window_size
    """
    # ref_status的格式
    # ref_status[(cluster_id, new_ref_id)] = {
    #     "created": current_window,
    #     "last_used": current_window,
    #     "expired": False
    # }

    time_each_insert = []

    # === Step 1: iterate over new streaming vectors ===
    for row, vector_data in updated_vectors.items():
        vector_i = Vector(vector_data)
        vector_objects[row] = vector_i
        cluster_to_insert = None
        cluster_to_insert_id = None
        bucket1 = None

        t1 = time.time()

        # === Step 2: use hash to locate candidate cluster ===
        if len(clusters) != 0 and base_R1 is not None:
            bucket1 = check_hash1_bucket(hash_table_1_Ref, vector_i, base_R1, 10, threshold)
            near_ref, near_slot, min_angle_R1 = find_nearest_reference(bucket1, hash_table_1_Ref, vector_i)

            if near_ref is not None:
                cluster_to_insert_id = near_ref[0]
                ref_id = near_ref[1]
                cluster_to_insert = clusters[cluster_to_insert_id]

                # #  Refresh last_used for matched reference
                # ref_key = (cluster_to_insert_id, ref_id)
                # if ref_key in ref_status and not ref_status[ref_key]["expired"]:  # ref还没过期
                #     ref_status[ref_key]["last_used"] = current_window

        # === Step 3: threshold check (only non-expired refs) ===
        INSERT = False
        ref_to_insert_id = None
        if cluster_to_insert is not None:
            valid_refs = []
            for ref_id, ref in enumerate(cluster_to_insert.references):
                key = (cluster_to_insert_id, ref_id)
                if key not in ref_status or ref_status[key]["expired"]:
                    continue
                valid_refs.append((ref_id, ref))

            for ref_id, ref in valid_refs:
                angle = compute_angle(ref.vector.data, vector_i.data)
                if angle < ref.threshold:  # 记录能返回的
                    INSERT = True
                    vector_i.update_angle(id(ref), angle)  # 维护新加入的向量和每个ref的夹角大小
                    vector_i.angle_with_ref1 = compute_angle(vector_i.data, cluster_to_insert.R1.vector.data)
                    # Refresh usage timestamp
                    ref_status[(cluster_to_insert_id, ref_to_insert_id)]["last_used"] = current_window

        # === Step 4️: insert success ===
        if INSERT:
            cluster_to_insert.add_vector(vector_i)
            cluster_to_insert.update_old_references(vector_i)
            max_angle_vector = cluster_to_insert.search_max_angle(vector_i)

            # Add new reference if direction sufficiently different
            new_ref = cluster_to_insert.add_new_reference(vector_i, max_angle_vector, 10)
            if new_ref is not None:
                new_ref_id = len(cluster_to_insert.references) - 1
                ref_status[(cluster_to_insert_id, new_ref_id)] = {
                    "created": current_window,
                    "last_used": current_window,
                    "expired": False
                }

            vector_to_cluster[row] = cluster_to_insert_id

        # === Step 5️: if not inserted, create new cluster ===
        else:
            new_cluster = Cluster(threshold)
            new_cluster.add_vector(vector_i)
            new_cluster.update_old_references(vector_i)
            clusters.append(new_cluster)
            new_cid = len(clusters) - 1
            vector_to_cluster[row] = new_cid

            # Initialize first reference timestamp
            ref_status[(new_cid, id(new_cluster.R1))] = {
                "created": current_window,
                "last_used": current_window,
                "expired": False
            }

            # Update base_R1 and hash tables
            if len(clusters) == 1 and base_R1 is None:
                base_R1 = new_cluster.R1
            if bucket1 is None:
                bucket1 = 0
            add_hash_ref1(hash_table_1_Ref, new_cluster.R1, bucket1, new_cid)

        t2 = time.time()
        time_each_insert.append(t2 - t1)

    # # === Step 6️: expire references after each window ===
    # ref_status = expire_references(ref_status, current_window, window_size)

    return (
        vector_objects,
        clusters,
        vector_to_cluster,
        time_each_insert,
        hash_table_1_Ref,
        base_R1,
        hash_table_2_Ref,
        proj_R,
        ref_status
    )


# === Reference expiration  ===
def expire_references(ref_status, current_window, window_size):
    """
    Mark references as expired if not used for more than window_size windows.
    """
    for key, info in ref_status.items():
        if not info["expired"] and (current_window - info["last_used"] > window_size):
            info["expired"] = True
            print("Expire Reference " + str(key))
    return ref_status



def Multiple_ref_clustering_streaming_Expire_Cluster(windows_updates, vectors, theta, window_num, window_size, para_theta):
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
    ref_status = {}

    for window_index, updates in windows_updates.items():
        if window_index >= window_num:
            break
        print(f"Processing updates for window {window_index}")
        updated_vectors = {}  # 在这个window变化的向量
        updated_vector_index = set()
        # 记录上一个没更新之前的向量
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
                # 如果变化前后的vector角度变化不大，只更新
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
        vector_objects, clusters, vector_to_cluster, time_each_insert, hash_table_1_Ref, base_R1, hash_table_2_Ref, proj_R, ref_status \
            = Multiple_ref_clustering_incremental_expire(
            vector_objects, updated_vectors, clusters, threshold, vector_to_cluster,
            hash_table_1_Ref, base_R1, hash_table_2_Ref, proj_R,
            window_index, ref_status)
        # === Step 6️: expire references after each window ===
        ref_status = expire_references(ref_status, window_index, window_size)
        t2 = time.time()
        print(t2 - t1)
        time_each_window.append(t2 - t1)
        time_full_insert += time_each_insert

    print("几个vector", len(vectors))
    num = 0
    for cluster in clusters:
        if len(cluster.vectors) > 0:
            num += 1
            print("几个ref", len(cluster.references), "几个vec", len(cluster.vectors))
    print("几个cluster", num)

    vectors_torch = [torch.tensor(vec, dtype=torch.float32) for vec in vectors]
    final_clusters = []
    for cluster in clusters:
        if len(cluster.vectors) > 0:
            vec_indices = [k for k, v in vector_objects.items() if v in cluster.vectors]
            final_clusters.append(vec_indices)

    return vectors_torch, final_clusters