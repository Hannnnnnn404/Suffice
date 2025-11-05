import time

import numpy as np
import math
from collections import defaultdict
import networkx as nx
import torch
from dataset.Slinding_window import process_streaming_data
from sklearn.neighbors import BallTree

import torch
from collections import defaultdict


def extract_torch_vectors_and_clusters_from_dstream(stream, vectors_np):
    """
    将 DStream 聚类器的结果转换为：
    - vectors_torch: List[torch.Tensor]  按索引排序的所有向量
    - clusters: List[List[int]]          每个聚类包含的向量索引

    参数：
    - stream: AngleGridDStream 实例
    - vectors_np: List[np.ndarray] 所有向量

    返回：
    - vectors_torch: List[torch.Tensor]
    - clusters: List[List[int]]
    """
    vector_to_gid = {}
    gid_to_indices = defaultdict(list)

    for idx, vec in enumerate(vectors_np):
        gid = stream._assign_grid(vec)
        if gid is not None:
            vector_to_gid[idx] = gid
            gid_to_indices[gid].append(idx)

    # 获取当前 grid 的连接图（聚类结果）
    active_clusters = stream.cluster()  # List[Set[gid]]
    clusters = []
    for cluster in active_clusters:
        merged = []
        for gid in cluster:
            merged.extend(gid_to_indices.get(gid, []))
        if merged:
            clusters.append(merged)

    # 构造 torch.Tensor 形式的 vectors（按索引顺序排列）
    vectors_torch = [torch.tensor(vec, dtype=torch.float32) for vec in vectors_np]

    return vectors_torch, clusters


def angular_distance(u, v):
    cosine = np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)
    return math.acos(cosine)


def sample_direction_centers(theta, dim=100, max_centers=100):
    centers = []
    first = np.random.randn(dim)
    first /= np.linalg.norm(first)
    centers.append(first)

    while len(centers) < max_centers:
        best_candidate = None
        max_min_angle = 0
        for _ in range(500):
            v = np.random.randn(dim)
            v /= np.linalg.norm(v)
            min_angle = min(angular_distance(v, c) for c in centers)
            if min_angle > max_min_angle:
                max_min_angle = min_angle
                best_candidate = v
        if max_min_angle >= theta:
            centers.append(best_candidate)
        else:
            break
    return np.array(centers)


def safe_sample_direction_centers(theta, dim=100, max_centers=100, max_attempts_per_center=1000):
    """
    加速版：带内积快速筛选，适合超高维空间，采样center。
    """
    centers = []

    first = np.random.randn(dim)
    first /= np.linalg.norm(first)
    centers.append(first)

    cos_theta = np.cos(theta)

    while len(centers) < max_centers:
        success = False
        for _ in range(max_attempts_per_center):
            candidate = np.random.randn(dim)
            candidate /= np.linalg.norm(candidate)

            # 先快速筛一遍：如果与某个center的内积 > cos(theta)，直接跳过
            passed = True
            for c in centers:
                inner_product = np.dot(candidate, c)
                if inner_product > cos_theta:
                    passed = False
                    break

            if passed:
                centers.append(candidate)
                success = True
                break

        if not success:
            print(f"警告: 第 {len(centers)} 个 center 超过最大尝试次数，停止采样。")
            break

    return np.array(centers)


# class AngleGridDStream:
#     def __init__(self, theta, decay_lambda, dense_thresh, sparse_thresh, dim):
#         self.theta = theta
#         self.decay_lambda = decay_lambda
#         self.dense_thresh = dense_thresh
#         self.sparse_thresh = sparse_thresh
#         self.time = 0
#         self.dim = dim
#
#         # 采样 centers
#         self.centers = safe_sample_direction_centers(theta, dim)
#         print(f"采样完成，生成 center 数量: {len(self.centers)}")
#
#         # 建立 BallTree 加速 assign_grid
#         self.tree = BallTree(self.centers, metric='euclidean')
#
#         self.grid_density = defaultdict(float)
#         self.grid_last_updated = {}
#
#     def _normalize(self, vec):
#         return vec / np.linalg.norm(vec)
#
#     def _assign_grid(self, vec):
#         vec = self._normalize(vec).reshape(1, -1)
#         dist, idx = self.tree.query(vec, k=1)
#         center_idx = idx[0][0]
#
#         # 检查是否真的在角度 theta 范围内
#         if angular_distance(vec.flatten(), self.centers[center_idx]) <= self.theta:
#             return center_idx
#         else:
#             return None
#
#     def _decay(self, t_now):
#         for gid in list(self.grid_density):
#             last_time = self.grid_last_updated.get(gid, 0)
#             decay_factor = 2 ** (-self.decay_lambda * (t_now - last_time))
#             self.grid_density[gid] *= decay_factor
#             self.grid_last_updated[gid] = t_now
#             if self.grid_density[gid] < self.sparse_thresh:
#                 del self.grid_density[gid]
#                 del self.grid_last_updated[gid]
#
#     def insert(self, vec, t_now):
#         self.time = t_now
#         self._decay(t_now)
#         gid = self._assign_grid(vec)
#         if gid is not None:
#             self.grid_density[gid] += 1
#             self.grid_last_updated[gid] = t_now
#
#     def cluster(self):
#         dense = {gid for gid, d in self.grid_density.items() if d >= self.dense_thresh}
#         transitional = {gid for gid, d in self.grid_density.items() if self.sparse_thresh < d < self.dense_thresh}
#         active = dense | transitional
#
#         G = nx.Graph()
#         for gid in active:
#             G.add_node(gid)
#         for gid1 in active:
#             for gid2 in active:
#                 if gid1 >= gid2:
#                     continue
#                 angle = angular_distance(self.centers[gid1], self.centers[gid2])
#                 if angle <= self.theta:
#                     G.add_edge(gid1, gid2)
#
#         clusters = list(nx.connected_components(G))
#         return clusters


class AngleGridDStream:
    def __init__(self, theta, decay_lambda, dense_thresh, sparse_thresh, dim):
        self.theta = theta
        self.decay_lambda = decay_lambda
        self.dense_thresh = dense_thresh
        self.sparse_thresh = sparse_thresh
        self.time = 0
        self.dim = dim
        # 采样 centers
        self.centers = safe_sample_direction_centers(theta, dim)
        print(f"采样完成，生成 center 数量: {len(self.centers)}")
        self.grid_density = defaultdict(float)
        self.grid_last_updated = {}

    def _normalize(self, vec):
        return vec / np.linalg.norm(vec)

    def _assign_grid(self, vec):
        vec = self._normalize(vec)  # 将向量归一化
        for i, center in enumerate(self.centers):  # 映射到一个方向网格（由 centers 定义）
            if angular_distance(vec, center) <= self.theta:
                return i
        return None

    def _decay(self, t_now):
        for gid in list(self.grid_density):
            last_time = self.grid_last_updated.get(gid, 0)
            decay_factor = 2 ** (-self.decay_lambda * (t_now - last_time))
            self.grid_density[gid] *= decay_factor
            self.grid_last_updated[gid] = t_now
            if self.grid_density[gid] < self.sparse_thresh:
                del self.grid_density[gid]
                del self.grid_last_updated[gid]

    def insert(self, vec, t_now):  # 新的数据点到来插入聚类
        self.time = t_now
        self._decay(t_now)  # 对所有格子执行时间衰减；
        gid = self._assign_grid(vec)
        if gid is not None:
            self.grid_density[gid] += 1  # 更新该网格的密度值（+1）
            self.grid_last_updated[gid] = t_now

    def cluster(self):  # 根据当前 grid 的密度和中心向量，构建一个基于余弦角度的图，并提取所有连接的 grid 组合作为聚类结果。
        # Classify grids
        dense = {gid for gid, d in self.grid_density.items() if d >= self.dense_thresh}
        transitional = {gid for gid, d in self.grid_density.items() if self.sparse_thresh < d < self.dense_thresh}
        active = dense | transitional

        # Build graph
        G = nx.Graph()
        for gid in active:
            G.add_node(gid)
        for gid1 in active:
            for gid2 in active:
                if gid1 >= gid2:
                    continue
                angle = angular_distance(self.centers[gid1], self.centers[gid2])
                if angle <= self.theta:
                    G.add_edge(gid1, gid2)

        clusters = list(nx.connected_components(G))
        return clusters


def DStream(windows_updates, vectors, theta, window_num):
    """
    对于每个Streaming vectors，执行完整的Naive插入和删除
    :param windows_updates:
    :return:
    """
    clusters = []  # 多个index list组成，每个list代表一个cluster
    time_each_window = []
    vector_to_cluster = {}  # 每个vector对应的cluster_id

    initial_start = time.time()
    # 再添加
    stream = AngleGridDStream(theta=np.deg2rad(theta), decay_lambda=0.001, dense_thresh=3, sparse_thresh=0.5,
                              dim=len(vectors[0]))
    initial_end = time.time()
    initial_time = initial_end - initial_start

    for window_index, updates in windows_updates.items():
        if window_index >= window_num:
            break
        print(f"Processing updates for window {window_index}")
        updated_vectors = {}  # 在这个window变化的向量
        updated_vector_index = set()
        # 记录上一个没更新之前的向量
        # vectors_last = vectors

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

        for i, v in updated_vectors.items():
            stream.insert(v, window_num)
        clusters = stream.cluster()
        t2 = time.time()
        print(t2 - t1)
        time_each_window.append(t2 - t1)

    print("几个vector", len(vectors))
    num = 0
    for cluster in clusters:
        if len(cluster) > 0:
            num += 1
            print("几个vector", len(cluster))
    print("几个cluster", num)
    return time_each_window, initial_time


def DStream_Cluster(windows_updates, vectors, theta, window_num):
    """
    对于每个Streaming vectors，执行完整的Naive插入和删除
    :param windows_updates:
    :return:
    """
    clusters = []  # 多个index list组成，每个list代表一个cluster
    time_each_window = []
    vector_to_cluster = {}  # 每个vector对应的cluster_id

    initial_start = time.time()
    # 再添加
    stream = AngleGridDStream(theta=np.deg2rad(theta), decay_lambda=0.01, dense_thresh=10, sparse_thresh=1,
                              dim=len(vectors[0]))
    initial_end = time.time()
    initial_time = initial_end - initial_start

    for window_index, updates in windows_updates.items():
        if window_index >= window_num:
            break
        print(f"Processing updates for window {window_index}")
        updated_vectors = {}  # 在这个window变化的向量
        updated_vector_index = set()
        # 记录上一个没更新之前的向量
        # vectors_last = vectors

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

        for i, v in updated_vectors.items():
            stream.insert(v, window_num)
        clusters = stream.cluster()
        t2 = time.time()
        print(t2 - t1)
        time_each_window.append(t2 - t1)

    print("几个vector", len(vectors))
    num = 0
    for cluster in clusters:
        if len(cluster) > 0:
            num += 1
            print("几个vector", len(cluster))
    print("几个cluster", num)
    vectors_dstream, clusters_dstream = extract_torch_vectors_and_clusters_from_dstream(stream, vectors)
    return vectors_dstream, clusters_dstream
