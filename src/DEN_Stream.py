import time
from dataset.Slinding_window import process_streaming_data
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict


def extract_torch_vectors_and_clusters_from_denstream(stream, vectors_np):
    """
    将 DenStream 聚类器的结果转换为标准格式：
    - vectors: List[torch.Tensor]  所有向量（按索引顺序）
    - clusters: List[List[int]]    每个聚类中的 vector 索引列表

    参数：
    - stream: AngleThresholdDenStream 或 AngleStrictDenStream 实例
    - vectors_np: 原始 numpy 向量列表

    返回：
    - vectors: List[torch.Tensor]
    - clusters: List[List[int]]
    """
    # 所有向量，转为 torch.Tensor 格式
    vectors = [torch.tensor(vec, dtype=torch.float32) for vec in vectors_np]

    # 获得 macro-clusters：List[List[MicroClusterIndex]]
    macro_clusters = stream.output_macro_clusters()

    # 构造 clusters: List[List[int]]，每个聚类包含的 vector 索引
    micro_clusters = stream.clusters
    clusters = []

    for macro in macro_clusters:
        cluster_indices = []
        for mc_id in macro:
            # 对于 AngleThresholdDenStream，我们只能估计重心最近点的索引
            center = micro_clusters[mc_id].center
            similarities = [F.cosine_similarity(center, vec, dim=0).item() for vec in vectors]
            most_similar_idx = int(np.argmax(similarities))
            cluster_indices.append(most_similar_idx)
        clusters.append(cluster_indices)

    return vectors, clusters

class AngleThresholdMicroCluster:
    def __init__(self, vector, timestamp, theta):
        self.N = 1.0
        self.LS = vector.clone()
        self.center = F.normalize(vector.clone(), dim=0)
        self.timestamp = timestamp
        self.weight = 1.0
        self.phi_max = 0.0  # 当前所有点与中心最大夹角
        self.theta = theta  # 目标最大两点夹角

    def decay(self, t_now, decay_lambda):
        delta = t_now - self.timestamp
        if delta > 0:
            decay_factor = 2 ** (-decay_lambda * delta)
            self.N *= decay_factor
            self.LS *= decay_factor
            self.weight *= decay_factor
            self.timestamp = t_now
            if self.N > 0:
                self.center = F.normalize(self.LS / self.N, dim=0)

    def angular_distance(self, x, y):
        x = F.normalize(x, dim=0)
        y = F.normalize(y, dim=0)
        cosine = F.cosine_similarity(x, y, dim=0).clamp(-1, 1)
        return torch.acos(cosine)

    def can_absorb(self, vector):
        angle_to_center = self.angular_distance(vector, self.center)
        # 当前允许吸收的新点角度阈值
        current_threshold = self.theta - self.phi_max
        return angle_to_center <= current_threshold

    def absorb(self, vector):
        angle_to_center = self.angular_distance(vector, self.center)
        self.phi_max = max(self.phi_max, angle_to_center.item())
        self.LS += vector
        self.N += 1.0
        self.center = F.normalize(self.LS / self.N, dim=0)


class AngleThresholdDenStream:
    def __init__(self, theta, decay_lambda=0.01, min_weight=1e-2):
        self.theta = theta
        self.decay_lambda = decay_lambda
        self.min_weight = min_weight
        self.clusters = []
        self.time = 0

    def insert(self, vec_np, t_now):
        self.time = t_now
        vec = F.normalize(torch.tensor(vec_np, dtype=torch.float32), dim=0)

        # 衰减和清理
        alive_clusters = []
        for cluster in self.clusters:
            cluster.decay(t_now, self.decay_lambda)
            if cluster.N >= self.min_weight:
                alive_clusters.append(cluster)
        self.clusters = alive_clusters

        # 尝试插入
        for cluster in self.clusters:
            if cluster.can_absorb(vec):
                cluster.absorb(vec)
                return

        # 创建新簇
        self.clusters.append(AngleThresholdMicroCluster(vec, t_now, self.theta))

    def get_cluster_centers(self):
        return [(idx, cluster.center.numpy(), cluster.N, cluster.phi_max) for idx, cluster in enumerate(self.clusters)]

    def output_macro_clusters(self):
        """最终宏聚类输出，基于 center 和 phi_max 判断两个簇是否能合并"""
        n = len(self.clusters)
        graph = defaultdict(list)

        for i in range(n):
            for j in range(i + 1, n):
                if self.can_merge(self.clusters[i], self.clusters[j]):
                    graph[i].append(j)
                    graph[j].append(i)

        visited = [False] * n
        macro_clusters = []

        for i in range(n):
            if not visited[i]:
                queue = [i]
                current_macro = []
                while queue:
                    node = queue.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    current_macro.append(node)
                    for neighbor in graph[node]:
                        if not visited[neighbor]:
                            queue.append(neighbor)
                macro_clusters.append(current_macro)

        return macro_clusters

    def can_merge(self, cluster1, cluster2):
        """严格保证合并后任意两点夹角 ≤ theta"""

        center_angle = self.angular_distance(cluster1.center, cluster2.center)
        combined_max_angle = max(cluster1.phi_max, cluster2.phi_max)

        # 合并后任意两点夹角不会超过
        return center_angle + combined_max_angle <= self.theta

    def angular_distance(self, x, y):
        x = F.normalize(x, dim=0)
        y = F.normalize(y, dim=0)
        cosine = F.cosine_similarity(x, y, dim=0).clamp(-1, 1)
        return torch.acos(cosine)


class AngleStrictMicroCluster:
    def __init__(self, vector, timestamp):
        self.N = 1.0
        self.LS = vector.clone()
        self.center = F.normalize(vector.clone(), dim=0)
        self.points = [vector.clone()]  # 保存所有向量
        self.timestamp = timestamp
        self.weight = 1.0

    def decay(self, t_now, decay_lambda):
        delta = t_now - self.timestamp
        if delta > 0:
            decay_factor = 2 ** (-decay_lambda * delta)
            self.N *= decay_factor
            self.LS *= decay_factor
            self.weight *= decay_factor
            self.timestamp = t_now
            if self.N > 0:
                self.center = F.normalize(self.LS / self.N, dim=0)

    def angular_distance(self, x, y):
        x = F.normalize(x, dim=0)
        y = F.normalize(y, dim=0)
        cosine = F.cosine_similarity(x, y, dim=0).clamp(-1, 1)
        return torch.acos(cosine)

    def can_absorb(self, vector, theta):
        for p in self.points:
            if self.angular_distance(vector, p) > theta:
                return False
        return True

    def absorb(self, vector):
        self.points.append(vector.clone())
        self.LS += vector
        self.N += 1.0
        self.center = F.normalize(self.LS / self.N, dim=0)


class AngleStrictDenStream:
    def __init__(self, theta, decay_lambda=0.01, min_weight=1e-2):
        self.theta = theta
        self.decay_lambda = decay_lambda
        self.min_weight = min_weight
        self.clusters = []
        self.time = 0

    def insert(self, vec_np, t_now):
        self.time = t_now
        vec = F.normalize(torch.tensor(vec_np, dtype=torch.float32), dim=0)

        # Apply decay and clean
        alive_clusters = []
        for cluster in self.clusters:
            cluster.decay(t_now, self.decay_lambda)
            if cluster.N >= self.min_weight:
                alive_clusters.append(cluster)
        self.clusters = alive_clusters

        # Try inserting into existing clusters
        for cluster in self.clusters:
            if cluster.can_absorb(vec, self.theta):
                cluster.absorb(vec)
                return

        # Otherwise, create a new cluster
        self.clusters.append(AngleStrictMicroCluster(vec, t_now))

    def output_macro_clusters(self):
        n = len(self.clusters)
        graph = defaultdict(list)

        for i in range(n):
            for j in range(i + 1, n):
                if self.can_merge(self.clusters[i], self.clusters[j]):
                    graph[i].append(j)
                    graph[j].append(i)

        visited = [False] * n
        macro_clusters = []

        for i in range(n):
            if not visited[i]:
                queue = [i]
                current_macro = []
                while queue:
                    node = queue.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    current_macro.append(node)
                    for neighbor in graph[node]:
                        if not visited[neighbor]:
                            queue.append(neighbor)
                macro_clusters.append(current_macro)

        return macro_clusters

    def can_merge(self, cluster1, cluster2):
        """严格判定：任意两点夹角 <= theta"""
        for p in cluster1.points:
            for q in cluster2.points:
                angle = self.angular_distance(p, q)
                if angle > self.theta:
                    return False
        return True

    def angular_distance(self, x, y):
        x = F.normalize(x, dim=0)
        y = F.normalize(y, dim=0)
        cosine = F.cosine_similarity(x, y, dim=0).clamp(-1, 1)
        return torch.acos(cosine)




def DENStream(windows_updates, vectors, theta, window_num):
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
    stream = AngleThresholdDenStream(theta=np.deg2rad(theta))
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
    return time_each_window, initial_time

def DENStream_Cluster(windows_updates, vectors, theta, window_num):
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
    stream = AngleThresholdDenStream(theta=np.deg2rad(theta))
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
