import time
import numpy as np
import math
import pandas as pd
import networkx as nx
import torch

from dataset.Slinding_window import process_streaming_data


def convert_links_clusters_to_standard_format(links_clusters):
    """
    将 Links 聚类的结果转换为标准格式：
    输入: links_clusters 是 List[Set[int]]
    输出: List[List[int]]
    """
    return [list(cluster) for cluster in links_clusters]


def evaluate_links_clustering(stream, clusters):
    vec_dict = stream.vector_store
    index_map = {}
    vectors = []

    for new_idx, (vec_id, vec) in enumerate(vec_dict.items()):
        index_map[vec_id] = new_idx
        vectors.append(vec)

    converted_clusters = [
        [index_map[vec_id] for vec_id in cluster] for cluster in clusters
    ]

    return converted_clusters


def cosine_similarity(u, v):
    return torch.dot(u, v) / (torch.norm(u) * torch.norm(v))


def angular_distance(u, v):
    cosine = torch.clamp(torch.dot(u, v) / (torch.norm(u) * torch.norm(v)), -1.0, 1.0)
    return torch.acos(cosine)


class Subcluster:
    def __init__(self, vec_id, vec):
        self.members = {vec_id}
        self.sum_vec = vec.clone()
        self.center = self.sum_vec / torch.norm(self.sum_vec)

    def add_member(self, vec_id, vec):
        self.members.add(vec_id)
        self.sum_vec = self.sum_vec + vec
        self.center = self.sum_vec / torch.norm(self.sum_vec)

    def remove_member(self, vec_id, vec):
        if vec_id in self.members:
            self.members.remove(vec_id)
            self.sum_vec = self.sum_vec - vec
            if len(self.members) > 0:
                self.center = self.sum_vec / torch.norm(self.sum_vec)


class LinksClusterFull:
    def __init__(self, Tc, Ts):
        self.Tc = Tc  # vector-subcluster threshold
        self.Ts = Ts  # subcluster-subcluster similarity threshold
        self.subclusters = []
        self.subcluster_graph = nx.Graph()
        self.vector_store = {}  # vec_id -> vec (torch.Tensor)
        self.vec_to_subcluster = {}  # vec_id -> subcluster index

    def insert_vector(self, vec_id, vec):
        vec = vec / torch.norm(vec)
        self.vector_store[vec_id] = vec

        best_sim = -1
        best_idx = -1
        for i, subc in enumerate(self.subclusters):
            sim = cosine_similarity(vec, subc.center)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= self.Tc:
            self.subclusters[best_idx].add_member(vec_id, vec)
            self.vec_to_subcluster[vec_id] = best_idx
            self.update_subcluster_edges(best_idx)
        else:
            new_subc = Subcluster(vec_id, vec)
            new_idx = len(self.subclusters)
            self.subclusters.append(new_subc)
            self.vec_to_subcluster[vec_id] = new_idx
            self.subcluster_graph.add_node(new_idx)
            self.update_subcluster_edges(new_idx)

    def update_subcluster_edges(self, idx):
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
        return [set.union(*(self.subclusters[i].members for i in cc))
                for cc in nx.connected_components(self.subcluster_graph)]


def LinksStream(windows_updates, vectors, Tc, Ts, window_num):
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
            stream.insert_vector(i, v)

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
    return time_each_window, initial_time


def LinksStream_Cluster(windows_updates, vectors, Tc, Ts, window_num):
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
            stream.insert_vector(i, v)

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
    standard_clusters = convert_links_clusters_to_standard_format(stream.get_clusters())
    converted_clusters = evaluate_links_clustering(stream, standard_clusters)

    return vectors, converted_clusters

