import torch
import numpy as np
from collections import defaultdict
from math import radians
import math
from collections import defaultdict
import networkx as nx
from dataset.Slinding_window import process_streaming_data


def extract_vectors_and_clusters_from_hsrc(stream):
    """
    从 HSRCCluster 中提取标准格式的 vectors 和 clusters。
    返回：
        vectors: List[torch.Tensor] 按索引排列
        clusters: List[List[int]] 每个列表是一个簇的向量索引
    """
    vectors_dict = stream.vectors  # {id: vector}
    vectors = [None] * (max(vectors_dict.keys()) + 1)
    for idx, vec in vectors_dict.items():
        vectors[idx] = vec
    clusters = [list(cluster) for cluster in stream.get_clusters()]
    return vectors, clusters


def angular_distance(u, v):
    cosine = torch.clamp(torch.dot(u, v) / (torch.norm(u) * torch.norm(v)), -1.0, 1.0)
    return torch.arccos(cosine)


def normalize(v):
    return v / torch.norm(v)


class HSRCCluster:
    def __init__(self, k, s, theta, dim, device='cpu'):
        self.k = k
        self.s = s
        self.theta = theta
        self.dim = dim
        self.device = device

        self.vectors = {}  # id -> vector (torch tensor)
        self.representatives = {}  # id -> vector
        self.rep_clusters = {}  # rep_id -> cluster_id
        self.assignments = {}  # vec_id -> rep_id
        self.clusters = defaultdict(set)  # cluster_id -> set of vec_ids

        self.cluster_id_counter = 0
        self.rep_graph = nx.Graph()

    def _build_knn_graph(self):
        ids = list(self.vectors.keys())
        if len(ids) <= self.k:
            self.knn = {vid: [] for vid in ids}
            return

        data = torch.stack([self.vectors[i] for i in ids])  # shape: (n, d)
        data = torch.nn.functional.normalize(data, p=2, dim=1)
        sim_matrix = torch.matmul(data, data.T)
        sim_matrix.fill_diagonal_(-float('inf'))
        topk_sim, topk_indices = torch.topk(sim_matrix, self.k, dim=1)
        self.knn = {ids[i]: [ids[j] for j in topk_indices[i]] for i in range(len(ids))}

    def _sample_representatives(self):
        all_ids = list(self.vectors.keys())
        sample_size = int(self.s * len(all_ids)) if self.s < 1 else int(self.s)
        selected_ids = np.random.choice(all_ids, sample_size, replace=False)
        for rid in selected_ids:
            self.representatives[rid] = self.vectors[rid]

    def _assign_non_reps(self):
        for vid, vec in self.vectors.items():
            if vid in self.representatives:
                continue
            best_sim = -1
            best_rep = None
            for rid, rvec in self.representatives.items():
                sim = torch.dot(vec, rvec)
                if sim > best_sim:
                    best_sim = sim
                    best_rep = rid
            self.assignments[vid] = best_rep

    def _build_rep_graph(self):
        self.rep_graph.clear()
        rep_ids = list(self.representatives.keys())
        for i in range(len(rep_ids)):
            for j in range(i + 1, len(rep_ids)):
                ri, rj = rep_ids[i], rep_ids[j]
                angle = angular_distance(self.representatives[ri], self.representatives[rj])
                if angle <= self.theta:
                    self.rep_graph.add_edge(ri, rj)

        for i, cc in enumerate(nx.connected_components(self.rep_graph)):
            for rid in cc:
                self.rep_clusters[rid] = i

    def _expand_clusters(self):
        self.clusters.clear()
        for vid, rep in self.assignments.items():
            cluster_id = self.rep_clusters.get(rep, None)
            if cluster_id is not None:
                self.clusters[cluster_id].add(vid)
        for rep, cid in self.rep_clusters.items():
            self.clusters[cid].add(rep)

    def _rebuild(self):
        self._build_knn_graph()
        self._sample_representatives()
        self._assign_non_reps()
        self._build_rep_graph()
        self._expand_clusters()

    def insert_vector(self, vec_id, vec_tensor):
        vec = normalize(vec_tensor.to(self.device))
        self.vectors[vec_id] = vec
        self._rebuild()

    def update_value(self, vec_id, dim_index, delta):
        if vec_id not in self.vectors:
            return
        self.vectors[vec_id][dim_index] += delta
        self.vectors[vec_id] = normalize(self.vectors[vec_id])
        self._rebuild()

    def update_vector(self, vec_id, new_vec_tensor):
        self.vectors[vec_id] = normalize(new_vec_tensor.to(self.device))
        self._rebuild()

    def update_dimension(self, vec_id, new_dim_tensor):
        self.vectors[vec_id] = normalize(new_dim_tensor.to(self.device))
        self._rebuild()

    def get_clusters(self):
        return list(self.clusters.values())


import time


def HSRCStream(windows_updates, vectors, theta, k, s, window_num, dim, device='cpu'):
    """
    HSRC 聚类流式处理主函数。
    :param windows_updates: dict[int, DataFrame] 每个窗口的增量更新
    :param vectors: List[Tensor] 当前所有向量
    :param theta: float 类中心夹角阈值
    :param k: int 最近邻数量
    :param s: float 代表点比例
    :param window_num: int 处理多少窗口
    :param dim: int 向量维度
    :param device: 'cpu' or 'cuda'
    """
    from collections import defaultdict
    clusters = []
    time_each_window = []

    initial_start = time.time()
    stream = HSRCCluster(k=k, s=s, theta=theta, dim=dim, device=device)
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

        # 全部重新构建向量（适配 HSRC 的 rebuild 特性）
        for row in updated_vectors:
            stream.update_vector(row, updated_vectors[row])

        clusters = stream.get_clusters()
        t2 = time.time()
        print("窗口时间:", t2 - t1)
        time_each_window.append(t2 - t1)

    print("全部vector数量:", len(vectors))
    print("最终聚类数:", len([c for c in clusters if len(c) > 0]))
    return time_each_window, initial_time


def HSRCStream_Cluster(windows_updates, vectors, theta, k, s, window_num, dim, device='cpu'):
    """
    HSRC 聚类流式处理主函数。
    :param windows_updates: dict[int, DataFrame] 每个窗口的增量更新
    :param vectors: List[Tensor] 当前所有向量
    :param theta: float 类中心夹角阈值
    :param k: int 最近邻数量
    :param s: float 代表点比例
    :param window_num: int 处理多少窗口
    :param dim: int 向量维度
    :param device: 'cpu' or 'cuda'
    """
    from collections import defaultdict
    clusters = []
    time_each_window = []

    initial_start = time.time()
    stream = HSRCCluster(k=k, s=s, theta=theta, dim=dim, device=device)
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

        # 全部重新构建向量（适配 HSRC 的 rebuild 特性）
        for row in updated_vectors:
            stream.update_vector(row, updated_vectors[row])

        clusters = stream.get_clusters()
        t2 = time.time()
        print("窗口时间:", t2 - t1)
        time_each_window.append(t2 - t1)

    print("全部vector数量:", len(vectors))
    print("最终聚类数:", len([c for c in clusters if len(c) > 0]))
    vectors_hsrc, clusters_hsrc = extract_vectors_and_clusters_from_hsrc(stream)
    return vectors_hsrc, clusters_hsrc
