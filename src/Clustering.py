import math
from collections import defaultdict

import torch
import numpy as np
import time


def cosine_similarity(v1, v2):
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))


def compute_angle(v1, v2):
    """计算两个向量的夹角"""
    cos_theta = cosine_similarity(v1, v2)
    return math.acos(max(-1, min(1, cos_theta)))  # 确保cos值合法


def add_hash_ref1(hash_table_1, R, bucket, c_id):
    """每个cluster的R1加入hash1, 已经存在一个cluster的R1了，以这个R1为基准"""
    hash_table_1[bucket].append((c_id, R))


def remove_hash_ref1(hash_table_1, R, bucket, c_id):
    hash_table_1[bucket].remove((c_id, R))


def check_hash1_bucket(hash_table_1, vector, base_R, ht1_size, threshold):
    angle = compute_angle(vector.data, base_R.vector.data)
    bucket = min(ht1_size - 1, int((angle / math.pi) * ht1_size))
    return bucket


def add_hash_ref2(hash_table_2, R, bucket, c_id):
    """每个cluster的R1加入hash1, 已经存在一个cluster的R1了，以这个R1为基准"""
    hash_table_2[bucket].append((c_id, R))


def remove_hash_ref2(hash_table_2, R, bucket, c_id):
    hash_table_2[bucket].remove((c_id, R))


def check_hash2_bucket(hash_table_2, vector, base_R, proj_R, ht2_size, threshold):
    proj = (torch.dot(vector.data, base_R.vector.data) / torch.dot(base_R.vector.data, base_R.vector.data)) * base_R.vector.data
    v_perp = proj - vector.data  # 垂直分量
    angle = compute_angle(v_perp, proj_R)
    bucket = min(ht2_size - 1, int((angle / math.pi) * ht2_size))
    return bucket


def find_nearest_reference(current_slot, hash_table_2, query_vector):
    total_slots = len(hash_table_2)
    checked = set()
    best_ref = None
    min_angle = float('inf')
    best_slot = None

    for offset in range(total_slots):
        for direction in [+1, -1]:
            slot = (current_slot + direction * offset) % total_slots
            if slot in checked:
                continue
            checked.add(slot)

            refs = hash_table_2[slot]
            for ref in refs:
                # 相邻桶号内多个ref
                angle = compute_angle(ref[1].vector.data, query_vector.data)
                if angle < min_angle:
                    min_angle = angle
                    best_ref = ref  # (0, Ref)
                    best_slot = slot

            if best_ref is not None:
                return best_ref, best_slot, min_angle  # 找到最近的就直接返回

    return None, None, None  # 所有桶都空

class Vector:
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.status = 'active'  # 默认状态为活跃
        self.angle_with_refs = {}  # 存储与参考向量的夹角
        self.angle_with_ref1 = 0  # 存储与参考向量1的夹角
        self.angle_with_proj = 0  # 储存与proj参考的夹角
        self.key1 = -1  # hash1的key
        self.key2 = -1  # hash2的key

    def update_angle(self, ref_id, angle):  # 更新向量和所有reference的夹角，此时已经选定一个cluster
        self.angle_with_refs[ref_id] = angle


class Reference:
    def __init__(self, vector, max_angle=0, threshold=0):
        self.vector = vector
        self.max_angle = max_angle
        self.threshold = threshold


class Cluster:
    def __init__(self, theta, ht1_size=10, ht2_size=18):
        self.vectors = []
        self.references = []
        self.ht1_size = ht1_size
        self.ht2_size = ht2_size
        self.hash_table_1 = {i: [] for i in range(ht1_size)}
        self.hash_table_2 = {i: [] for i in range(ht2_size)}
        self.threshold = theta
        self.R1 = None  # reference类
        self.v_perp0 = None  # tensor

    def add_vector(self, vector):
        # 实现向量的加入
        self.vectors.append(vector)
        if self.R1 is None:  # 如果是第一个vector，赋值给R1
            self.R1 = Reference(vector, max_angle=0, threshold=self.threshold)
            self.references.append(self.R1)
            vector.angle_with_refs[id(self.R1)] = 0

        if self.v_perp0 is None and vector != self.R1.vector and self.R1 is not None:  # 如果是第二个vector，计算norm base
            # 计算 v2 在 R1 上的投影
            proj_R1_v2 = (torch.dot(vector.data, self.R1.vector.data) / torch.dot(self.R1.vector.data,
                                                                                  self.R1.vector.data)) * self.R1.vector.data
            self.v_perp0 = proj_R1_v2 - vector.data  # 作为参考方向

        # 加入hash table
        self.add_hash_table1(vector)
        self.add_hash_table2(vector)

    def delete_vector(self, vector):
        """
        从cluster中删除vector
        :param vector:
        :return:
        """
        if vector in self.vectors:
            self.vectors.remove(vector)
            # 从hash table中移除

    def update_old_references(self, vector):
        """
        更新旧参考向量的和新加入的vector的max angle和阈值，只看新的参考向量对他是否有影响
        :param vector:
        :param angle: angle between vectors and old references
        :return:
        """
        for ref in self.references:
            # print("update old reference", id(ref), "\n")
            if vector.angle_with_refs[id(ref)] > ref.max_angle:
                ref.max_angle = vector.angle_with_refs[id(ref)]
                ref.threshold = self.threshold - ref.max_angle

    def add_new_reference(self, vector, max_angle, para_num_overlap):
        """
        vector是否能成为新的参考向量: no coverage/full coverage/no coverage
        :param para_num_overlap: overlap的次数
        :param max_angle: 新的参考向量的max angle，之前必然搜索过
        :param vector:
        :return: 成为新的参考向量返回新的参考向量，不成为返回false
        """
        No_coverage = 0
        Full_coverage = False
        Overlap = 0
        overlap_area = 0

        t_new = self.threshold - max_angle  # 新vector成为ref的threshold
        for ref in self.references:
            delta = vector.angle_with_refs[id(ref)]  # 两个ref的夹角
            t_old = ref.threshold
            angle_overlap = t_old + t_new - delta
            if t_old + t_new <= delta:  # no coverage
                No_coverage += 1
            elif t_old >= delta + t_new:  # full coverage
                Full_coverage = True
                return False
            else:  # overlap，同时计算overlap的面积
                Overlap += 1
                overlap_area += 1 - math.cos(angle_overlap)

            if Full_coverage:  # 不需要添加新的ref
                return False
            if No_coverage == len(self.references):  # 全部ref都no coverage，产生新reference
                new_ref = Reference(vector, max_angle, t_new)
                self.references.append(new_ref)
                # print("Add new ref")
                return new_ref
            if Overlap >= para_num_overlap:  # 太多的overlap，不必产生新的ref
                return False
            new_ref_area = 1 - math.cos(t_new)
            if overlap_area >= new_ref_area:  # 太多的overlap，不必产生新的ref
                return False

            # print("Add new ref")
            new_ref = Reference(vector, max_angle, t_new)  # 少量的overlap，可以产生新的ref
            self.references.append(new_ref)
            return new_ref

    def check_cluster_to_insert(self, vector):  # 暂时先比较所有cluster内的vector
        """
        检查向量能加入到哪个现有聚类，返回和现有聚类的最小夹角
        :param vector
        :return:
         - 返回和现有聚类的最小夹角
        """
        min_difference_ht2 = abs(self.R1.vector.key2 - vector.key2)
        min_difference_ht1 = abs(self.R1.vector.key1 - vector.key1)
        return min_difference_ht2, min_difference_ht1

    def check_threshold_to_insert_cluster(self, vector):
        """
        判断向量是否能加入聚类
        :param vector
        :return:
          - 如果能加入这个聚类，则返回加入的reference和角度，同时存储向量到每个参考向量的夹角
          - 如果不能加入这个聚类，返回False
        """
        Join_cluster = False
        Join_ref = None
        Join_angle = None
        vector.angle_with_ref1 = compute_angle(vector.data, self.R1.vector.data)
        for ref in self.references:
            # print("check threshold with ref ", id(ref), "\n")
            angle = compute_angle(vector.data, ref.vector.data)
            vector.update_angle(id(ref), angle)  # 维护新加入的向量和每个ref的夹角大小
            if angle < ref.threshold:  # 记录能返回的
                Join_cluster = True
                Join_ref = ref
                Join_angle = angle
        if Join_cluster:
            return Join_cluster, Join_ref, Join_angle
        vector.angle_with_ref1 = 0  # 不能加入这个聚类，还原vector
        vector.angle_with_refs = {}
        return False, None, None

    def add_hash_table1(self, vector):  # 加入之前已经计算过vector和R1的夹角
        """
        将vector加入hash table1, hash function = [beta/theta]*size
        :param vector: 要加入的vector
        :return:
        """
        if vector.angle_with_ref1 == 0:
            vector.angle_with_ref1 = compute_angle(vector.data, self.R1.vector.data)
        bucket = min(self.ht1_size - 1, int((vector.angle_with_ref1 / self.threshold) * self.ht1_size))
        vector.key1 = bucket
        self.hash_table_1[bucket].append(vector)

    def add_hash_table2(self, vector):
        """
        将vector加入hash table2, hash function = [angle/pi]*size
        :param vector: 要加入的vector
        :return:
        """
        # 计算 vector 在 R1 上的投影
        proj_R1 = (torch.dot(vector.data, self.R1.vector.data) / torch.dot(self.R1.vector.data,
                                                                           self.R1.vector.data)) * self.R1.vector.data
        v_perp = proj_R1 - vector.data  # 垂直分量

        # 计算 v_perp 和 v_perp0 的夹角
        beta = 0
        if self.v_perp0 is not None:
            beta = compute_angle(v_perp, self.v_perp0)
        vector.angle_with_proj = beta

        # 哈希到 HT2
        bucket = min(self.ht2_size - 1, int((beta / math.pi) * self.ht2_size))
        vector.key2 = bucket
        self.hash_table_2[bucket].append(vector)

    def remove_hash_table1(self, vector):
        """
        将vector从hash table1中移除
        :param vector:
        :return:
        """
        self.hash_table_1[vector.key1].remove(vector)
        vector.key1 = -1
        vector.angle_with_ref1 = 0

    def remove_hash_table2(self, vector):
        """
        将vector从hash table2中移除
        :param vector:
        :return:
        """
        self.hash_table_2[vector.key2].remove(vector)
        vector.key2 = -1
        vector.angle_with_proj = 0

    def search_max_angle(self, vector):
        """
        搜索vector的最大angle
        :param vector:
        :return:返回最大angle
        """

        max_angle = vector.angle_with_ref1  # 维护 max angle
        # 在 HT2 中找到最远的 bucket
        ht2_bucket = vector.key2
        far_buckets = self.ht2_size - ht2_bucket - 1  # 从远到近搜索
        visited_h2 = []  # 存储访问过的bucket
        search_h2_index_1 = -1
        search_h2_index_2 = -1
        search_h2_tuple = ()  # 搜索次远的方向的ht2 bucket
        count_direction = 0  # 通过次数判断是否在同侧
        direction_para = math.floor(self.ht2_size / 2) - 1

        while True:
            if search_h2_tuple == ():
                search_h2_tuple = (far_buckets, far_buckets)
                search_h2_index_1 = far_buckets
                search_h2_index_2 = far_buckets
            elif search_h2_tuple[0] >= self.ht2_size - 1 or search_h2_tuple[1] <= 0:
                break
            else:  # 计算下两个次远的ht2 bucket，由于对称所以会有两个
                count_direction += 1
                search_h2_index_1 += 1
                search_h2_index_2 -= 1
                search_h2_a = search_h2_index_1
                search_h2_b = search_h2_index_2
                if search_h2_a > self.ht2_size - 1:
                    search_h2_a = self.ht2_size - (search_h2_a - self.ht2_size + 1)
                if search_h2_b < 0:
                    search_h2_b = -search_h2_b - 1
                search_h2_tuple = (search_h2_a, search_h2_b)

            for search_h2_bucket in search_h2_tuple:
                if search_h2_bucket in visited_h2:
                    continue

                # 如果当前搜索的ht2_bucket和vector在同侧，直接估计最大值，然后结束
                if count_direction >= direction_para:
                    # 获取所有桶号的集合
                    all_buckets = set(self.hash_table_2.keys())
                    # 获取未访问的桶号通过差集
                    unvisited_buckets = all_buckets - set(visited_h2)
                    max_angle_with_ref1 = 0
                    for i in unvisited_buckets:
                        unvisited_vector = self.hash_table_2[i]  # 为访问的vector
                        if len(unvisited_vector) > 0:
                            max_angle_with_ref1 = max(unvisited_vector, key=lambda x: x.angle_with_ref1)
                        # for v in unvisited_vector:
                        #     if v.angle_with_ref1 > max_angle_with_ref1:
                        #         max_angle_with_ref1 = v.angle_with_ref1

                    max_angle_direction = 0
                    if max_angle_with_ref1 != 0:
                        max_angle_direction = vector.angle_with_ref1 + max_angle_with_ref1.angle_with_ref1
                    if max_angle_direction >= max_angle:
                        max_angle = max_angle_direction
                    return max_angle

                # 如果当前搜索的ht2_bucket和vector在反侧, 继续搜索
                search_h2 = self.hash_table_2[search_h2_bucket]  # 存储某方向的所有vector
                visited_h2.append(search_h2_bucket)
                search_h1 = defaultdict(list)  # 存储这些向量和对应在HT1的桶号:vector字典

                for v in search_h2:  # 对这个ht2桶的几个向量
                    search_h1[v.key1].append(v)  # 找ht1的编号
                sorted_ht1_keys = sorted(search_h1.keys(), reverse=True)  # HT1 桶号从大到小排序

                for ht1_bucket in sorted_ht1_keys:
                    if len(search_h1[ht1_bucket]) <= 5:  # 如果这个桶内的向量很少,直接比不求上界了
                        for v_j in search_h1[ht1_bucket]:
                            angle = compute_angle(v_j.data, vector.data)
                            if max_angle > angle:
                                max_angle = angle

                    else:
                        # 剪枝优化：如果当前 bucket 最大可能夹角 ≤ max_angle，跳过这个桶的搜索
                        beta_r2 = vector.angle_with_ref1  # vector和R1的夹角
                        theta_r2 = vector.angle_with_proj  # vector和proj base的夹角
                        bucket_angle_upper_bound = self.compute_vj_vi_upper_bound(ht1_bucket, search_h2_bucket,
                                                                                  beta_r2,
                                                                                  theta_r2)  # 计算当前要搜索的ht1 bucket的上界夹角
                        if bucket_angle_upper_bound <= max_angle:
                            break

                        for v_j in search_h1[ht1_bucket]:  # 该桶内的向量
                            max_angle = compute_angle(vector.data, v_j.data)

        return max_angle

    def compute_vj_vi_upper_bound(self, ht1_bucket, ht2_bucket, beta_r2, theta_r2):
        """
        计算 v_j 和 r2 之间的夹角上界
        ht1_bucket: v_j 在 HT1 的桶编号
        ht2_bucket: v_j 在 HT2 的桶编号
        ht1_size: HT1 的总桶数
        ht2_size: HT2 的总桶数
        beta_r2: r2 的 β 角度
        theta_r2: r2 的 θ 角度
        """
        # 计算 v_j 的最大 β
        beta_max = (ht1_bucket + 1) * (self.threshold / self.ht1_size)

        # 计算 v_j 的 θ_j 上界（选择最远的端点）
        theta_upper = (ht2_bucket + 1) * (math.pi / self.ht2_size)
        theta_lower = ht2_bucket * (math.pi / self.ht2_size)

        # 计算两种情况，选择最大的夹角
        cos_theta_max_1 = (math.cos(beta_max) * math.cos(beta_r2) +
                           math.sin(beta_max) * math.sin(beta_r2) * math.cos(theta_upper - theta_r2))

        cos_theta_max_2 = (math.cos(beta_max) * math.cos(beta_r2) +
                           math.sin(beta_max) * math.sin(beta_r2) * math.cos(theta_lower - theta_r2))

        # 取最小的 cos 值（最大夹角）
        cos_theta_max = min(cos_theta_max_1, cos_theta_max_2)

        # 计算最大夹角
        theta_max = math.acos(min(1.0, max(-1.0, cos_theta_max)))

        return theta_max
