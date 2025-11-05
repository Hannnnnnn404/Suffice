import numpy as np
import math
import time
from collections import defaultdict
import networkx as nx
from collections import defaultdict
import torch
from dataset.Slinding_window import process_streaming_data


def extract_torch_vectors_and_clusters_from_euclidean_dstream(stream, vectors_np):
    """
    将 EuclideanGridDStream 聚类器的结果转换为：
    - vectors_torch: List[torch.Tensor]  按索引排序的所有向量
    - clusters: List[List[int]]          每个聚类包含的向量索引

    参数：
    - stream: EuclideanGridDStream 实例
    - vectors_np: List[np.ndarray] 所有向量 (原始 numpy 格式)

    返回：
    - vectors_torch: List[torch.Tensor]
    - clusters: List[List[int]]
    """

    # Step 1️: 构建 “grid -> vector indices” 映射
    grid_to_indices = defaultdict(list)
    vector_to_grid = {}

    for idx, vec in enumerate(vectors_np):
        gid = stream._assign_grid(vec)
        vector_to_grid[idx] = gid
        grid_to_indices[gid].append(idx)

    # Step 2️: 获取聚类（grid 集合）
    active_clusters = stream.cluster()  # list[set[grid_coord]]

    # Step 3️: 合并每个 cluster 对应的所有 vector index
    clusters = []
    for cluster_grids in active_clusters:
        merged_indices = []
        for gid in cluster_grids:
            merged_indices.extend(grid_to_indices.get(gid, []))
        if merged_indices:
            clusters.append(merged_indices)

    # Step 4️: 构造 torch.Tensor 形式的 vectors
    vectors_torch = [torch.tensor(vec, dtype=torch.float32) for vec in vectors_np]

    return vectors_torch, clusters


class EuclideanGridDStream:
    def __init__(self, decay_lambda, dense_thresh, sparse_thresh, dim, grid_width=0.1):
        """
        原始 D-Stream 算法的欧几里得版本
        :param decay_lambda: 衰减系数
        :param dense_thresh: 密度阈值（dense）
        :param sparse_thresh: 密度阈值（sparse）
        :param dim: 向量维度
        :param grid_width: 网格划分宽度
        """
        self.decay_lambda = decay_lambda
        self.dense_thresh = dense_thresh
        self.sparse_thresh = sparse_thresh
        self.time = 0
        self.dim = dim
        self.grid_width = grid_width

        # 每个grid由其离散坐标（tuple）表示
        self.grid_density = defaultdict(float)
        self.grid_last_updated = {}

    def _assign_grid(self, vec):
        """
        将向量分配到欧几里得网格，自动兼容 numpy 和 torch 输入
        """
        # 若是 torch.Tensor，则转 numpy
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()

        grid_coord = tuple((vec // self.grid_width).astype(int))
        return grid_coord

    def _decay(self, t_now):
        """
        对所有 grid 执行时间衰减
        """
        for gid in list(self.grid_density):
            last_time = self.grid_last_updated.get(gid, 0)
            decay_factor = 2 ** (-self.decay_lambda * (t_now - last_time))
            self.grid_density[gid] *= decay_factor
            self.grid_last_updated[gid] = t_now
            if self.grid_density[gid] < self.sparse_thresh:
                del self.grid_density[gid]
                del self.grid_last_updated[gid]

    def insert(self, vec, t_now):
        """
        插入新数据点
        """
        self.time = t_now
        self._decay(t_now)
        gid = self._assign_grid(vec)
        self.grid_density[gid] += 1
        self.grid_last_updated[gid] = t_now

    def _neighbor_grids(self, gid):
        """
        找到与 gid 相邻的所有网格坐标
        """
        neighbors = []
        for offset in np.ndindex(*(3,) * self.dim):
            if all(o == 1 for o in offset):
                continue
            delta = np.array(offset) - 1
            neighbor = tuple(np.array(gid) + delta)
            neighbors.append(neighbor)
        return neighbors

    def cluster(self):  # 直接基于欧几里得距离判断邻近（无需显式生成邻居坐标）。
        dense = {gid for gid, d in self.grid_density.items() if d >= self.dense_thresh}
        transitional = {gid for gid, d in self.grid_density.items() if self.sparse_thresh < d < self.dense_thresh}
        active = list(dense | transitional)

        G = nx.Graph()
        for gid in active:
            G.add_node(gid)

        active_coords = np.array([np.array(g) for g in active])
        for i, g1 in enumerate(active):
            for j in range(i + 1, len(active)):
                dist = np.linalg.norm(active_coords[i] - active_coords[j])
                if dist <= 1.5:  # 根据 grid_width 定义邻接阈值
                    G.add_edge(g1, active[j])

        clusters = list(nx.connected_components(G))
        return clusters


def DStream_Original_Cluster(windows_updates, vectors, window_num):
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
    stream = EuclideanGridDStream(decay_lambda=0.01, dense_thresh=10, sparse_thresh=1, dim=len(vectors[0]))
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
    vectors_dstream, clusters_dstream = extract_torch_vectors_and_clusters_from_euclidean_dstream(stream, vectors)
    return vectors_dstream, clusters_dstream

