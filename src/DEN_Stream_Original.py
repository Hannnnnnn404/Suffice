import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import math
import time
from dataset.Slinding_window import process_streaming_data

def extract_torch_vectors_and_clusters_from_denstream(stream, vectors_np):
    """
    适配 EuclideanDenStream：
    - vectors: List[torch.Tensor]
    - clusters: List[List[int]]  （每个宏簇→用其包含的微簇中心，匹配到全局向量中“最近”的样本索引做代表；与原角度版保持同样输出规格）
    """
    vectors = [torch.tensor(vec, dtype=torch.float32) for vec in vectors_np]

    macro_clusters = stream.output_macro_clusters()  # List[List[p_mc_idx]]
    clusters = []

    # 取 p-mc 列表
    micro_clusters = stream.p_clusters

    for macro in macro_clusters:
        cluster_indices = []
        for mc_id in macro:
            center = torch.tensor(micro_clusters[mc_id].center, dtype=torch.float32)
            # 找到距离中心最近的一个样本索引（代表性取样，与你现有角度版行为一致）
            dists = [torch.norm(center - v).item() for v in vectors]
            nearest_idx = int(np.argmin(dists))
            cluster_indices.append(nearest_idx)
        clusters.append(cluster_indices)

    return vectors, clusters

class EuclideanMicroCluster:
    def __init__(self, vec_np, timestamp):
        # 统一到 numpy，内部计算用 numpy，输出/比较时可再转 torch
        if isinstance(vec_np, torch.Tensor):
            vec_np = vec_np.detach().cpu().numpy()
        self.N = 1.0
        self.LS = vec_np.astype(np.float32).copy()
        self.SS = np.dot(vec_np, vec_np).item()  # 标量
        self.center = self.LS / self.N
        self.timestamp = timestamp
        self.weight = 1.0  # 与 N 等价，用于接口一致性

    def decay(self, t_now, decay_lambda):
        delta = t_now - self.timestamp
        if delta > 0:
            factor = 2 ** (-decay_lambda * delta)
            self.N *= factor
            self.LS *= factor
            self.SS *= factor
            self.weight *= factor
            self.timestamp = t_now
            if self.N > 0:
                self.center = self.LS / self.N

    def radius(self):
        # 经验半径（可用于监控）：sqrt( SS/N - ||center||^2 )
        if self.N <= 0:
            return 0.0
        return math.sqrt(max(self.SS / self.N - np.dot(self.center, self.center), 0.0))

    def dist_to_center(self, vec_np):
        if isinstance(vec_np, torch.Tensor):
            vec_np = vec_np.detach().cpu().numpy()
        diff = vec_np - self.center
        return np.linalg.norm(diff)

    def absorb(self, vec_np):
        if isinstance(vec_np, torch.Tensor):
            vec_np = vec_np.detach().cpu().numpy()
        self.LS += vec_np
        self.SS += float(np.dot(vec_np, vec_np))
        self.N += 1.0
        self.weight = self.N
        self.center = self.LS / self.N



class EuclideanDenStream:
    """
    原始 DenStream 的欧氏实现（简化且稳定）：
    - 欧氏半径 eps 决定吸收
    - β·μ 决定 o-mc 升级到 p-mc
    - min_weight 用于清理弱簇（衰减后）
    - 宏簇：对 p-mc 的中心按欧氏距离阈值 eps_merge 连边取连通分量
    """
    def __init__(self, eps, beta=0.2, mu=5.0, decay_lambda=0.01,
                 min_weight=1e-2, eps_merge=None):
        self.eps = float(eps)
        self.beta = float(beta)
        self.mu = float(mu)
        self.decay_lambda = float(decay_lambda)
        self.min_weight = float(min_weight)
        self.eps_merge = float(eps_merge) if eps_merge is not None else float(eps)

        self.p_clusters = []  # potential micro-clusters
        self.o_clusters = []  # outlier micro-clusters
        self.time = 0

    # ---- 维护 ----
    def _decay_and_prune(self, t_now):
        alive_p, alive_o = [], []
        for mc in self.p_clusters:
            mc.decay(t_now, self.decay_lambda)
            if mc.N >= self.min_weight:
                alive_p.append(mc)
        for mc in self.o_clusters:
            mc.decay(t_now, self.decay_lambda)
            if mc.N >= self.min_weight:
                alive_o.append(mc)
        self.p_clusters, self.o_clusters = alive_p, alive_o

    def _promote_if_needed(self, mc):
        if mc.N >= self.beta * self.mu and mc in self.o_clusters:
            # 提升为 p-mc
            self.o_clusters.remove(mc)
            self.p_clusters.append(mc)

    # ---- 插入 ----
    def insert(self, vec_np, t_now):
        self.time = t_now
        # 统一 numpy
        if isinstance(vec_np, torch.Tensor):
            vec_np = vec_np.detach().cpu().numpy()

        # 衰减+清理
        self._decay_and_prune(t_now)

        # 1) 尝试吸收到最近的 p-mc（欧氏）
        best_p, best_p_dist = None, float('inf')
        for mc in self.p_clusters:
            d = mc.dist_to_center(vec_np)
            if d < best_p_dist:
                best_p, best_p_dist = mc, d
        if best_p is not None and best_p_dist <= self.eps:
            best_p.absorb(vec_np)
            return

        # 2) 尝试吸收到最近的 o-mc
        best_o, best_o_dist = None, float('inf')
        for mc in self.o_clusters:
            d = mc.dist_to_center(vec_np)
            if d < best_o_dist:
                best_o, best_o_dist = mc, d
        if best_o is not None and best_o_dist <= self.eps:
            best_o.absorb(vec_np)
            self._promote_if_needed(best_o)
            return

        # 3) 否则创建新的 o-mc
        new_mc = EuclideanMicroCluster(vec_np, t_now)
        self.o_clusters.append(new_mc)
        self._promote_if_needed(new_mc)

    # ---- 监控 ----
    def get_cluster_centers(self):
        """
        返回所有微簇的信息（仅 p-mc，便于和你原打印逻辑对应）
        列表项：(idx, center(np.ndarray), weight, radius)
        """
        out = []
        for idx, mc in enumerate(self.p_clusters):
            out.append((idx, mc.center.copy(), mc.N, mc.radius()))
        return out

    # ---- 宏簇输出（只基于 p-mc）----
    def output_macro_clusters(self):
        """
        返回 List[List[int]]，每个列表是 p-mc 的索引集合（连通分量）
        """
        n = len(self.p_clusters)
        if n == 0:
            return []

        # 构建邻接：中心间欧氏距离 <= eps_merge
        adj = [[] for _ in range(n)]
        centers = np.stack([mc.center for mc in self.p_clusters], axis=0)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(centers[i] - centers[j]) <= self.eps_merge:
                    adj[i].append(j)
                    adj[j].append(i)

        # 连通分量
        visited = [False] * n
        macros = []
        for i in range(n):
            if not visited[i]:
                q = [i]
                comp = []
                while q:
                    u = q.pop()
                    if visited[u]:
                        continue
                    visited[u] = True
                    comp.append(u)
                    for v in adj[u]:
                        if not visited[v]:
                            q.append(v)
                macros.append(comp)
        return macros


def DENStream_Original_Cluster(windows_updates, vectors, window_num):
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
    # 例如：eps=2.0，beta=0.2，mu=5.0 可作为起点
    stream = EuclideanDenStream(eps=2.0, beta=0.2, mu=5.0, decay_lambda=0.01, min_weight=1e-2, eps_merge=2.0)
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
        # clusters = stream.get_cluster_centers()
        # # 输出 macro-clusters
        # macro_clusters = stream.output_macro_clusters()
        # print(f"宏簇数量: {len(macro_clusters)}")
        # for i, macro in enumerate(macro_clusters):
        #     print(f"Macro-cluster {i}: 包含微簇 {macro}")

        t2 = time.time()
        print(t2 - t1)
        time_each_window.append(t2 - t1)

    print("几个vector", len(vectors))
    # 查看当前每个簇中心和点数
    for idx, (center_id, center_vec, weight, phi_max) in enumerate(stream.get_cluster_centers()):
        print(f"Cluster {center_id}: size = {weight:.2f}, max_angle = {np.rad2deg(phi_max):.2f} degrees")
    vectors_den, clusters_den = extract_torch_vectors_and_clusters_from_denstream(stream, vectors)
    return vectors_den, clusters_den
