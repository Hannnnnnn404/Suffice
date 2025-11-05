import time
import numpy as np
import math
import pandas as pd
import networkx as nx
import torch
from dataset.Slinding_window import process_streaming_data


def cosine_similarity(u, v):
    return torch.dot(u, v) / (torch.norm(u) * torch.norm(v))


class Subcluster:
    """
    一个子簇（Subcluster）代表一组方向相似的向量。
    每个子簇维护：
        - members: 向量ID集合
        - sum_vec: 所有向量的加和
        - center: 单位化后的方向中心
        - weight: 成员数量或衰减后的权重
        - last_update: 最近更新时间戳
    """
    def __init__(self, vec_id, vec, timestamp):
        self.members = {vec_id}
        self.sum_vec = vec.clone()
        self.center = self.sum_vec / torch.norm(self.sum_vec)
        self.weight = 1.0
        self.last_update = timestamp

    def add_member(self, vec_id, vec, timestamp):
        self.members.add(vec_id)
        self.sum_vec += vec
        self.center = self.sum_vec / torch.norm(self.sum_vec)
        self.weight += 1.0
        self.last_update = timestamp

    def remove_member(self, vec_id, vec):
        if vec_id in self.members:
            self.members.remove(vec_id)
            self.sum_vec -= vec
            if len(self.members) > 0:
                self.center = self.sum_vec / torch.norm(self.sum_vec)
            self.weight = max(len(self.members), 1.0)

    def decay(self, t_now, decay_lambda):
        """指数衰减，表示旧数据权重降低"""
        delta_t = t_now - self.last_update
        if delta_t > 0:
            decay_factor = 2 ** (-decay_lambda * delta_t)
            self.weight *= decay_factor
            self.sum_vec *= decay_factor
            if torch.norm(self.sum_vec) > 0:
                self.center = self.sum_vec / torch.norm(self.sum_vec)
            self.last_update = t_now


class LinksClusterFull:
    """
    Links: High-dimensional Online Clustering (Mansfield et al., 2018)
    聚类定义：
        - Subcluster 内部：方向相似（cosine ≥ Tc）
        - Subcluster 之间：中心相似（cosine ≥ Ts）
        - Cluster = 连通的 subcluster 分量
    """
    def __init__(self, Tc, Ts, decay_lambda=0.0, min_weight=1e-3):
        self.Tc = Tc  # 向量吸收阈值
        self.Ts = Ts  # subcluster 相似阈值
        self.decay_lambda = decay_lambda
        self.min_weight = min_weight
        self.subclusters = []
        self.subcluster_graph = nx.Graph()
        self.vector_store = {}  # vec_id -> vec
        self.vec_to_subcluster = {}  # vec_id -> subcluster idx
        self.time = 0

    def insert_vector(self, vec_id, vec_np, t_now):
        """插入新向量并维护结构"""
        self.time = t_now
        vec = torch.tensor(vec_np, dtype=torch.float32)
        vec = vec / torch.norm(vec)
        self.vector_store[vec_id] = vec

        # Step 1: 所有子簇权重衰减 + 清理
        alive_subclusters = []
        for subc in self.subclusters:
            subc.decay(t_now, self.decay_lambda)
            if subc.weight >= self.min_weight:
                alive_subclusters.append(subc)
        self.subclusters = alive_subclusters

        # Step 2: 找到最相似的subcluster
        best_sim, best_idx = -1, -1
        for i, subc in enumerate(self.subclusters):
            sim = cosine_similarity(vec, subc.center)
            if sim > best_sim:
                best_sim, best_idx = sim, i

        # Step 3: 判断是否吸收或新建
        if best_sim >= self.Tc:
            self.subclusters[best_idx].add_member(vec_id, vec, t_now)
            self.vec_to_subcluster[vec_id] = best_idx
            self.update_subcluster_edges(best_idx)
        else:
            new_subc = Subcluster(vec_id, vec, t_now)
            new_idx = len(self.subclusters)
            self.subclusters.append(new_subc)
            self.vec_to_subcluster[vec_id] = new_idx
            self.subcluster_graph.add_node(new_idx)
            self.update_subcluster_edges(new_idx)

    def update_subcluster_edges(self, idx):
        """更新 subcluster 图：相似度≥Ts则连边"""
        for j, subc in enumerate(self.subclusters):
            if j == idx:
                continue
            sim = cosine_similarity(self.subclusters[idx].center, subc.center)
            if sim >= self.Ts:
                self.subcluster_graph.add_edge(idx, j)
            else:
                if self.subcluster_graph.has_edge(idx, j):
                    self.subcluster_graph.remove_edge(idx, j)

    def delete_vector(self, vec_id):
        """删除过期向量"""
        if vec_id not in self.vec_to_subcluster:
            return
        idx = self.vec_to_subcluster[vec_id]
        vec = self.vector_store[vec_id]
        self.subclusters[idx].remove_member(vec_id, vec)
        del self.vector_store[vec_id]
        del self.vec_to_subcluster[vec_id]
        if len(self.subclusters[idx].members) == 0:
            if self.subcluster_graph.has_node(idx):
                self.subcluster_graph.remove_node(idx)
        else:
            self.update_subcluster_edges(idx)

    def get_clusters(self):
        """输出宏簇（每个连通分量即一个cluster）"""
        clusters = []
        for component in nx.connected_components(self.subcluster_graph):
            members = set()
            for i in component:
                members.update(self.subclusters[i].members)
            clusters.append(members)
        return clusters


# --------------------- Streaming Driver ---------------------
def LinksStream_Original_Cluster(windows_updates, vectors, Tc, Ts, window_num):
    clusters = []
    time_each_window = []

    initial_start = time.time()
    stream = LinksClusterFull(Tc=Tc, Ts=Ts)
    initial_end = time.time()
    initial_time = initial_end - initial_start

    for window_index, updates in windows_updates.items():
        if window_index >= window_num:
            break
        print(f"Processing updates for window {window_index}")
        updated_vectors = {}
        updated_vector_index = set()

        for _, update in updates.iterrows():
            row = update['row_index']
            col = update['col_index']
            behave = update['behave']

            if behave == 1:
                vectors[row][col] += 1
            elif behave == -1:
                vectors[row][col] -= 1

            updated_vector_index.add(row)

        t1 = time.time()
        for row in updated_vector_index:
            updated_vectors[row] = vectors[row]

        for row in updated_vectors:
            stream.delete_vector(row)

        for i, v in updated_vectors.items():
            stream.insert_vector(i, v, window_index)

        clusters = stream.get_clusters()
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

    return [torch.tensor(v, dtype=torch.float32) for v in vectors], [list(c) for c in clusters]
