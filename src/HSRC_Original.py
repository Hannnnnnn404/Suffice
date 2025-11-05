import torch
import numpy as np
import networkx as nx
import time
from collections import defaultdict
from dataset.Slinding_window import process_streaming_data
import math


def extract_vectors_and_clusters_from_hsrc(stream):
    """
    从 HSRCCluster 中提取标准格式的 vectors 和 clusters。
    返回：
        vectors: List[torch.Tensor]
        clusters: List[List[int]]
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


class HSRCClusterOriginal:
    def __init__(self, s, theta, dim, device='cpu'):
        """
        原始 HSRC 算法
        :param s: float, 代表点比例 (0,1]
        :param theta: float, 角度阈值 (弧度)
        :param dim: 向量维度
        """
        self.s = s
        self.theta = theta
        self.dim = dim
        self.device = device

        # 向量存储
        self.vectors = {}  # id -> vector
        # 代表点
        self.representatives = {}
        # 非代表点 -> 对应代表点
        self.assignments = {}
        # cluster_id -> set of vec_ids
        self.clusters = defaultdict(set)

        self.cluster_id_counter = 0
        self.rep_graph = nx.Graph()

    # ------------------- Step 1. 重采样代表点 -------------------
    def _sample_representatives(self):
        all_ids = list(self.vectors.keys())
        if len(all_ids) == 0:
            return
        sample_size = int(self.s * len(all_ids)) if self.s < 1 else int(self.s)
        sample_size = max(1, min(sample_size, len(all_ids)))
        selected_ids = np.random.choice(all_ids, sample_size, replace=False)
        self.representatives = {rid: self.vectors[rid] for rid in selected_ids}

    # ------------------- Step 2. 构建代表点图 -------------------
    def _build_rep_graph(self):
        """
        原始 HSRC 的关键步骤：
        只检查代表点之间的夹角是否 ≤ θ，
        若是，则连边。最终每个连通分量即为一个 cluster。
        """
        self.rep_graph.clear()
        rep_ids = list(self.representatives.keys())
        for i in range(len(rep_ids)):
            for j in range(i + 1, len(rep_ids)):
                ri, rj = rep_ids[i], rep_ids[j]
                angle = angular_distance(self.representatives[ri], self.representatives[rj])
                if angle <= self.theta:
                    self.rep_graph.add_edge(ri, rj)

        # 每个连通分量 => cluster_id
        self.rep_clusters = {}
        for i, cc in enumerate(nx.connected_components(self.rep_graph)):
            for rid in cc:
                self.rep_clusters[rid] = i

    # ------------------- Step 3. 非代表点分配 -------------------
    def _assign_non_reps(self):
        """
        每个非代表点分配给相似度最高的代表点，
        并继承该代表点所在的 cluster。
        """
        self.assignments.clear()
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

    # ------------------- Step 4. 聚类扩展 -------------------
    def _expand_clusters(self):
        self.clusters.clear()
        # 将代表点加入各自簇
        for rid, cid in self.rep_clusters.items():
            self.clusters[cid].add(rid)
        # 将非代表点加入代表点的簇
        for vid, rid in self.assignments.items():
            cid = self.rep_clusters.get(rid, None)
            if cid is not None:
                self.clusters[cid].add(vid)

    # ------------------- Step 5. 重新构建 -------------------
    def _rebuild(self):
        """
        原始 HSRC 重建逻辑：
        每次更新后重新采样代表点 -> 构建代表点图 -> 分配非代表点 -> 聚类扩展
        """
        if len(self.vectors) == 0:
            return
        self._sample_representatives()
        self._build_rep_graph()
        self._assign_non_reps()
        self._expand_clusters()

    # ------------------- 插入 / 更新 -------------------
    def insert_vector(self, vec_id, vec_tensor):
        vec = normalize(vec_tensor.to(self.device))
        self.vectors[vec_id] = vec
        self._rebuild()

    def update_vector(self, vec_id, new_vec_tensor):
        self.vectors[vec_id] = normalize(new_vec_tensor.to(self.device))
        self._rebuild()

    def get_clusters(self):
        return list(self.clusters.values())


# ===============================================================
# 流式接口
# ===============================================================
def HSRCStream_Original_Cluster(windows_updates, vectors, theta, s, window_num, dim, device='cpu'):
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
    stream = HSRCClusterOriginal(s=s, theta=theta, dim=dim, device=device)
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
