import os
import pdb
from tqdm import tqdm
import time

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from scipy.spatial import cKDTree
import open3d as o3d
# import faiss
import numba as nb
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
# Set CUDA environment variable to ensure synchronous execution for debugging
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

@nb.njit(parallel=True)
def compute_densities(lims, D, radius):
    N = len(lims) - 1
    densities = np.zeros(N, dtype=np.float32)
    sigma2 = (radius/2)**2
    for i in nb.prange(N):
        start, end = lims[i], lims[i+1]
        if end > start:
            dists = np.sqrt(D[start:end])
            weights = np.exp(-dists ** 2 / (2 * sigma2))
            densities[i] = weights.sum()
        else:
            densities[i] = 0.0
    return densities


class Quantize_kMeans():
    def __init__(self, num_clusters=100, num_iters=10):
        self.num_clusters = num_clusters
        self.num_kmeans_iters = num_iters
        self.nn_index = torch.empty(0)
        self.centers = torch.empty(0)
        self.vec_dim = 0
        self.cluster_ids = torch.empty(0)
        self.cls_ids = torch.empty(0)
        self.excl_clusters = []
        self.excl_cluster_ids = []
        self.cluster_len = torch.empty(0)
        self.max_cnt = 0
        self.n_excl_cls = 0

        self.weights = 0
        self.cweights = None  # Placeholder for cluster weights, if needed

    def get_weights(self, gaussian):
        chunk_size = 1500
        dynamic_range = 10.0
        positions = gaussian._xyz.to('cuda')
        N = positions.shape[0]
        densities = torch.zeros(N, device='cuda', dtype=torch.float32)
        
        std = torch.std(positions, dim=0).mean()
        sigma = (0.2 * std).item()
        bbox = positions.max(dim=0)[0] - positions.min(dim=0)[0]
        radius = (bbox.mean() * 0.05).item()  # 5% scale of the bounding box
        start = time.time()
        for i in range(0, N, chunk_size):
            chunk = positions[i:i+chunk_size]
            # distance
            dists = torch.cdist(chunk, positions)  # [chunk, N]
            mask = (dists < radius).float()
            weights = torch.exp(-dists ** 2 / (2 * sigma ** 2)) * mask
            densities[i:i+chunk.shape[0]] = weights.sum(dim=1)
            del dists, mask, weights
            torch.cuda.empty_cache()
        # 对数归一化，拉平极端分布
        densities = torch.log1p(densities)
        # 归一化到[0,1]
        densities = (densities - densities.min()) / (densities.max() - densities.min() + 1e-6)
        # 控制最大/最小权重比
        weights = densities * (dynamic_range - 1) / dynamic_range + 1.0 / dynamic_range
        # 再归一化到[0,1]
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
        weights = weights.half()
        print("shape of weights:", weights.shape)
        print(f"权重范围: {weights.min().item():.4f} ~ {weights.max().item():.4f}")
        print(f"计算权重耗时: {time.time() - start:.2f}秒")
        return weights
    def get_weights_(self, gaussian, compu=False, dynamic_range=10.0, neighbor_ratio=0.05):
        positions = gaussian._xyz.detach().cpu().numpy()  # [N, 3]
        N = positions.shape[0]
        densities = np.zeros(N, dtype=np.float32)
        # 计算邻域半径
        bbox = positions.max(axis=0) - positions.min(axis=0)
        radius = bbox.mean() * neighbor_ratio
        # 构建KDTree
        start = time.time()
        tree = cKDTree(positions)
        # 查询每个点的邻域点
        for i in range(N):
            idx = tree.query_ball_point(positions[i], r=radius)
            # 只和邻域点算距离
            dists = np.linalg.norm(positions[idx] - positions[i], axis=1)
            weights = np.exp(-dists ** 2 / (2 * (radius/2)**2))
            densities[i] = weights.sum()
        # 对数归一化
        densities = np.log1p(densities)
        densities = (densities - densities.min()) / (densities.max() - densities.min() + 1e-6)
        weights = densities * (dynamic_range - 1) / dynamic_range + 1.0 / dynamic_range
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
        weights = torch.from_numpy(weights).half().cuda()
        print("shape of weights:", weights.shape)
        print(f"权重范围: {weights.min().item():.4f} ~ {weights.max().item():.4f}")
        print(f"计算权重耗时: {time.time() - start:.2f}秒")
        return weights
    
    def get_weights_open3d(self, gaussian, dynamic_range=10.0, neighbor_ratio=0.05):
        positions = gaussian._xyz.detach().cpu().numpy()
        N = positions.shape[0]
        bbox = positions.max(axis=0) - positions.min(axis=0)
        radius = bbox.mean() * neighbor_ratio
        start = time.time()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        densities = np.zeros(N, dtype=np.float32)
        for i in range(N):
            [_, idx, dists] = kdtree.search_radius_vector_3d(pcd.points[i], radius)
            dists = np.sqrt(dists)
            weights = np.exp(-dists ** 2 / (2 * (radius/2)**2))
            densities[i] = weights.sum()

        # 后续归一化同前
        densities = np.log1p(densities)
        densities = (densities - densities.min()) / (densities.max() - densities.min() + 1e-6)
        weights = densities * (dynamic_range - 1) / dynamic_range + 1.0 / dynamic_range
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
        weights = torch.from_numpy(weights).half().cuda()
        print("shape of weights:", weights.shape)
        print(f"权重范围: {weights.min().item():.4f} ~ {weights.max().item():.4f}")
        print(f"计算权重耗时: {time.time() - start:.2f}秒")
        return weights

    def get_weights_faiss(self, gaussian, dynamic_range=10.0, neighbor_ratio=0.1, device=0):
        """
        用faiss GPU实现高斯点密度权重计算，适合百万级点云。
        """
        positions = gaussian._xyz.detach().cpu().numpy().astype('float32')  # [N, 3]
        N = positions.shape[0]
        bbox = positions.max(axis=0) - positions.min(axis=0)
        radius = bbox.mean() * neighbor_ratio
        start = time.time()

        # 方案2：如果要用GPU，可以用IVF索引
        quantizer = faiss.IndexFlatL2(3)
        index = faiss.IndexIVFFlat(quantizer, 3, min(100, N//10))  # nlist设为N//10或100
        index.train(positions)
        index.add(positions)
        index.nprobe = 10  # 搜索的聚类数

        # faiss range search
        lims, D, I = index.range_search(positions, radius**2)  # D: 距离的平方
        densities = np.zeros(N, dtype=np.float32)
        print(f"faiss计算权重耗时1: {time.time() - start:.2f}秒")
        
        densities = compute_densities(lims, D, radius)

        # 对数归一化、动态范围控制
        densities = np.log1p(densities)
        densities = (densities - densities.min()) / (densities.max() - densities.min() + 1e-6)
        weights = densities * (dynamic_range - 1) / dynamic_range + 1.0 / dynamic_range
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)

        self.weights = weights
        print(f"faiss计算权重耗时2: {time.time() - start:.2f}秒")
        print("shape of weights:", weights.shape)
        print(f"权重范围: {weights.min().item():.4f} ~ {weights.max().item():.4f}")

        plt.figure()
        plt.hist(densities, bins=100)
        plt.title('Density Distribution (Actual)')
        plt.xlabel('Density')
        plt.ylabel('Count')
        plt.savefig('density_hist.png')

        # 密度logscale直方图
        plt.figure()
        plt.hist(np.log(densities + 1e-8), bins=100)
        plt.title('Density Distribution (Log Scale)')
        plt.xlabel('log(Density)')
        plt.ylabel('Count')
        plt.savefig('logdensity_hist.png')

        # 权重直方图
        plt.figure()
        plt.hist(weights, bins=100)
        plt.title('Weight Distribution')
        plt.xlabel('Weight')
        plt.ylabel('Count')
        plt.savefig('w_hist.png')

        # # 停止程序
        # sys.exit()
        # weights = 1-weights
        weights = np.maximum(weights, 0.90)
        weights = torch.from_numpy(weights).half().cuda()
        self.weights = weights
        return weights

    # def get_weights_faiss(self, gaussian, dynamic_range=10.0, neighbor_ratio=0.1, device=0):
    #     """
    #     用faiss GPU实现高斯点密度权重计算，适合百万级点云。
    #     """
    #     positions = gaussian._xyz.detach().cpu().numpy().astype('float32')  # [N, 3]
    #     N = positions.shape[0]
    #     bbox = positions.max(axis=0) - positions.min(axis=0)
    #     radius = bbox.mean() * neighbor_ratio
    #     start = time.time()

    #     # 构建faiss GPU index
    #     # res = faiss.StandardGpuResources()

    #     # config = faiss.GpuIndexFlatConfig()
    #     # config.device = device
    #     # index = faiss.GpuIndexFlatL2(res, 3, config)

    #     # index.add(positions)
    #     # 使用faiss的CPU索引
    #     index = faiss.IndexFlatL2(3)  # 3是特征维度
    #     index.add(positions)

    #     # faiss range search
    #     lims, D, I = index.range_search(positions, radius**2)  # D: 距离的平方
    #     densities = np.zeros(N, dtype=np.float32)
    #     print(f"faiss计算权重耗时1: {time.time() - start:.2f}秒")
    #     # for i in range(N):
    #     #     start, end = lims[i], lims[i+1]
    #     #     if end > start:
    #     #         dists = np.sqrt(D[start:end])
    #     #         weights = np.exp(-dists ** 2 / (2 * (radius/2)**2))
    #     #         densities[i] = weights.sum()
    #     #     else:
    #     #         densities[i] = 0.0
    #     densities = compute_densities(lims, D, radius)

    #     # 对数归一化、动态范围控制
    #     densities = np.log1p(densities)
    #     densities = (densities - densities.min()) / (densities.max() - densities.min() + 1e-6)
    #     weights = densities * (dynamic_range - 1) / dynamic_range + 1.0 / dynamic_range
    #     weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
    #     # plt.figure()
    #     # plt.hist(densities, bins=100)
    #     # plt.title('Density Distribution (Actual)')
    #     # plt.xlabel('Density')
    #     # plt.ylabel('Count')
    #     # plt.show()

    #     # # 密度logscale直方图
    #     # plt.figure()
    #     # plt.hist(np.log(densities + 1e-8), bins=100)
    #     # plt.title('Density Distribution (Log Scale)')
    #     # plt.xlabel('log(Density)')
    #     # plt.ylabel('Count')
    #     # plt.show()

    #     # # 权重直方图
    #     # plt.figure()
    #     # plt.hist(weights, bins=100)
    #     # plt.title('Weight Distribution')
    #     # plt.xlabel('Weight')
    #     # plt.ylabel('Count')
    #     # plt.show()

    #     # # 停止程序
    #     # sys.exit()
    #     weights = torch.from_numpy(weights).half().cuda()

    #     self.weights = weights
    #     print(f"faiss计算权重耗时2: {time.time() - start:.2f}秒")
    #     print("shape of weights:", weights.shape)
    #     print(f"权重范围: {weights.min().item():.4f} ~ {weights.max().item():.4f}")
        
    #     # Show distribution of weights

    #     # plt.figure(figsize=(8, 5))
    #     # plt.hist(weights.cpu().numpy(), bins=50, color='royalblue', alpha=0.7)
    #     # plt.xlabel('Weight Value')
    #     # plt.ylabel('Count')
    #     # plt.title('Distribution of Gaussian Weights')
    #     # plt.grid(True)
    #     # plt.tight_layout()
    #     # # Save the histogram
    #     # plt.savefig('weights_hist.png')
    #     # print("已保存权重分布直方图为 weights_hist.png")
    #     # # Stop the script after saving the histogram
    #     # sys.exit(0)
    #     return weights
    
    def get_weights_near(self, gaussian, k=50):
        """计算每个点的局部密度"""
        # 计算每个点到k个最近邻居的平均距离
        points = gaussian._xyz.detach().cpu().numpy().astype('float32')
        nn = NearestNeighbors(n_neighbors=k+1)
        nn.fit(points)
        distances, _ = nn.kneighbors(points)
        
        # # 排除自身距离(索引0)，计算平均距离
        # mean_distances = np.mean(distances[:, 1:], axis=1)
        sigma = np.std(distances[:, 1:])
        print(f"std of distances: {sigma:.4f}")
        weights = np.exp(-distances[:, 1:]**2 / (2 * sigma**2))
        denom = np.sum(weights, axis=1)
        numer = np.sum(distances[:, 1:] * weights, axis=1)
        print(np.any(denom == 0))  # 如果为True，说明有分母为0
        print(np.isnan(distances).any(), np.isnan(weights).any())
        print(np.isinf(distances).any(), np.isinf(weights).any())

        mean_distances = np.zeros_like(numer)
        nonzero_mask = denom != 0
        mean_distances[nonzero_mask] = numer[nonzero_mask] / denom[nonzero_mask]
        mean_distances[~nonzero_mask] = mean_distances[nonzero_mask].max()  
        # mean_distances = np.sum(distances[:, 1:] * weights, axis=1) / np.sum(weights, axis=1) # Sometimes denom is zero

        
        # # original weights calculation
        # densities = 1 / (mean_distances + 1e-6)
        # dynamic_range = 10.0  # 动态范围
        # # densities = np.log1p(density)
        # densities = (densities - densities.min()) / (densities.max() - densities.min() + 1e-6)
        # weights = densities * (dynamic_range - 1) / dynamic_range + 1.0 / dynamic_range
        # weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
        # weights = 1 - weights  # 反转权重，使得密度高的点权重低
        # weights = np.maximum(weights, 0.5)

        weights = np.zeros(len(mean_distances))
        d_min = mean_distances.min()
        d_max = mean_distances.max()
        p99 = np.percentile(mean_distances, 99)
        mask1 = mean_distances <= p99
        mask2 = mean_distances > p99
        weights[mask1] = 1 - 0.10 * (p99 - mean_distances[mask1]) / (p99 - d_min)
        weights[mask2] = 1

        plt.figure(figsize=(8, 5))
        plt.hist(weights, bins=200, color='royalblue', alpha=0.7)
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
        plt.title('Distribution of Gaussian Weights')
        plt.grid(True)
        plt.tight_layout()
        # Save the histogram
        plt.savefig('weights_hist.png')
        print("已保存权重分布直方图为 weights0618.png")
        weights = torch.from_numpy(weights).half().cuda()
        self.weights = weights
        return weights

    def get_dist(self, x, y, mode='sq_euclidean'):
        """Calculate distance between all vectors in x and all vectors in y.

        x: (m, dim)
        y: (n, dim)
        dist: (m, n)
        """
        if mode == 'sq_euclidean_chunk':
            step = 65536
            if x.shape[0] < step:
                step = x.shape[0]
            dist = []
            for i in range(np.ceil(x.shape[0] / step).astype(int)):
                dist.append(torch.cdist(x[(i*step): (i+1)*step, :].unsqueeze(0), y.unsqueeze(0))[0])
            dist = torch.cat(dist, 0)
        elif mode == 'sq_euclidean':
            dist = torch.cdist(x.unsqueeze(0).detach(), y.unsqueeze(0).detach())[0]
        return dist

    # Update centers in non-cluster assignment iters using cached nn indices.
    def update_centers(self, feat):
        cweights = self.cweights 
        feat = feat.detach().reshape(-1, self.vec_dim)
        # Update all clusters except the excluded ones in a single operation
        # Add a dummy element with zeros at the end
        feat = torch.cat([feat, torch.zeros_like(feat[:1]).cuda()], 0)
        if cweights:
            weights = torch.cat([self.weights, torch.zeros(1, device=self.weights.device)])
            feat = feat * weights.view(-1, 1)
        self.centers = torch.sum(feat[self.cluster_ids, :].reshape(
            self.num_clusters, self.max_cnt, -1), dim=1)
        if len(self.excl_cluster_ids) > 0:
            for i, cls in enumerate(self.excl_clusters):
                # Division by num_points in cluster is done during the one-shot averaging of all
                # clusters below. Only the extra elements in the bigger clusters are added here.
                self.centers[cls] += torch.sum(feat[self.excl_cluster_ids[i], :], dim=0)
        self.centers /= (self.cluster_len + 1e-6)

    # def update_centers2(self, feat):
    #     cweights = self.cweights 
    #     feat = feat.detach().reshape(-1, self.vec_dim)
    #     # Update all clusters except the excluded ones in a single operation
    #     # Add a dummy element with zeros at the end
    #     feat = torch.cat([feat, torch.zeros_like(feat[:1]).cuda()], 0)
    #     if cweights:
    #         weights = torch.cat([self.weights, torch.zeros(1, device=self.weights.device)])
    #         feat = feat * weights.view(-1, 1)
    #     self.centers = torch.sum(feat[self.cluster_ids, :].reshape(
    #         self.num_clusters, self.max_cnt, -1), dim=1)
    #     if len(self.excl_cluster_ids) > 0:
    #         for i, cls in enumerate(self.excl_clusters):
    #             # Division by num_points in cluster is done during the one-shot averaging of all
    #             # clusters below. Only the extra elements in the bigger clusters are added here.
    #             self.centers[cls] += torch.sum(feat[self.excl_cluster_ids[i], :], dim=0)
    #     self.centers /= (self.cluster_len + 1e-6)

    # Update centers during cluster assignment using mask matrix multiplication
    # Mask is obtained from distance matrix
    def update_centers_(self, feat, cluster_mask=None, nn_index=None, avg=False):
        # cweights = self.cweights 
        feat = feat.detach().reshape(-1, self.vec_dim)
        # if cweights:
        #    feat = feat * self.weights.view(-1, 1)
        centers = (cluster_mask.T @ feat)
        if avg:
            self.centers /= counts.unsqueeze(-1)
        return centers

    def equalize_cluster_size(self):
        """Make the size of all the clusters the same by appending dummy elements.

        """
        # Find the maximum number of elements in a cluster, make size of all clusters
        # equal by appending dummy elements until size is equal to size of max cluster.
        # If max is too large, exclude it and consider the next biggest. Use for loop for
        # the excluded clusters and a single operation for the remaining ones for
        # updating the cluster centers.

        unq, n_unq = torch.unique(self.nn_index, return_counts=True)
        # Find max cluster size and exclude clusters greater than a threshold
        topk = 100
        if len(n_unq) < topk:
            topk = len(n_unq)
        max_cnt_topk, topk_idx = torch.topk(n_unq, topk)
        self.max_cnt = max_cnt_topk[0]
        idx = 0
        self.excl_clusters = []
        self.excl_cluster_ids = []
        while(self.max_cnt > 5000):
            self.excl_clusters.append(unq[topk_idx[idx]])
            idx += 1
            if idx < topk:
                self.max_cnt = max_cnt_topk[idx]
            else:
                break
        self.n_excl_cls = len(self.excl_clusters)
        self.excl_clusters = sorted(self.excl_clusters)
        # Store the indices of elements for each cluster
        all_ids = []
        cls_len = []
        for i in range(self.num_clusters):
            cur_cluster_ids = torch.where(self.nn_index == i)[0]
            # For excluded clusters, use only the first max_cnt elements
            # for averaging along with other clusters. Separately average the
            # remaining elements just for the excluded clusters.
            cls_len.append(torch.Tensor([len(cur_cluster_ids)]))
            if i in self.excl_clusters:
                self.excl_cluster_ids.append(cur_cluster_ids[self.max_cnt:])
                cur_cluster_ids = cur_cluster_ids[:self.max_cnt]
            # Append dummy elements to have same size for all clusters
            all_ids.append(torch.cat([cur_cluster_ids, -1 * torch.ones((self.max_cnt - len(cur_cluster_ids)),
                                                                       dtype=torch.long).cuda()]))
        all_ids = torch.cat(all_ids).type(torch.long)
        cls_len = torch.cat(cls_len).type(torch.long)
        self.cluster_ids = all_ids
        self.cluster_len = cls_len.unsqueeze(1).cuda()
        self.cls_ids = self.nn_index

    def cluster_assign(self, feat, feat_scaled=None):
        cweights = self.cweights 
        # quantize with kmeans
        feat = feat.detach()
        feat = feat.reshape(-1, self.vec_dim)
        if cweights:
            feat = feat * self.weights.view(-1, 1)
        if feat_scaled is None:
            feat_scaled = feat
            scale = feat[0] / (feat_scaled[0] + 1e-8)
        if len(self.centers) == 0:
            self.centers = feat[torch.randperm(feat.shape[0])[:self.num_clusters], :]

        # start kmeans
        chunk = True
        counts = torch.zeros(self.num_clusters, dtype=torch.float32).cuda() + 1e-6
        centers = torch.zeros_like(self.centers)
        for iteration in range(self.num_kmeans_iters):
            # chunk for memory issues
            if chunk:
                self.nn_index = None
                i = 0
                chunk = 10000
                while True:

                    dist = self.get_dist(feat[i*chunk:(i+1)*chunk, :], self.centers)
                    curr_nn_index = torch.argmin(dist, dim=-1)
                    # Assign a single cluster when distance to multiple clusters is same
                    dist = F.one_hot(curr_nn_index, self.num_clusters).type(torch.float32)
                    curr_centers = self.update_centers_(feat[i*chunk:(i+1)*chunk, :], dist, curr_nn_index, avg=False)
                    counts += dist.detach().sum(0) + 1e-6
                    centers += curr_centers
                    if self.nn_index == None:
                        self.nn_index = curr_nn_index
                    else:
                        self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
                    i += 1
                    if i*chunk > feat.shape[0]:
                        break

            self.centers = centers / counts.unsqueeze(-1)
            # Reinitialize to 0
            centers[centers != 0] = 0.
            counts[counts > 0.1] = 0.

        if chunk:
            self.nn_index = None
            i = 0
            # chunk = 100000
            while True:
                dist = self.get_dist(feat_scaled[i * chunk:(i + 1) * chunk, :], self.centers)
                curr_nn_index = torch.argmin(dist, dim=-1)
                if self.nn_index == None:
                    self.nn_index = curr_nn_index
                else:
                    self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
                i += 1
                if i * chunk > feat.shape[0]:
                    break
        self.equalize_cluster_size()

    # def cluster_assign2(self, feat, feat_scaled=None):
    #     cweights = self.cweights 
    #     # quantize with kmeans
    #     feat = feat.detach()
    #     feat = feat.reshape(-1, self.vec_dim)
    #     if cweights:
    #         feat = feat * self.weights.view(-1, 1)
    #     if feat_scaled is None:
    #         feat_scaled = feat
    #         scale = feat[0] / (feat_scaled[0] + 1e-8)
    #     if len(self.centers) == 0:
    #         self.centers = feat[torch.randperm(feat.shape[0])[:self.num_clusters], :]

    #     # start kmeans
    #     chunk = True
    #     counts = torch.zeros(self.num_clusters, dtype=torch.float32).cuda() + 1e-6
    #     centers = torch.zeros_like(self.centers)
    #     for iteration in range(self.num_kmeans_iters):
    #         # chunk for memory issues
    #         if chunk:
    #             self.nn_index = None
    #             i = 0
    #             chunk = 10000
    #             while True:

    #                 dist = self.get_dist(feat[i*chunk:(i+1)*chunk, :], self.centers)
    #                 curr_nn_index = torch.argmin(dist, dim=-1)
    #                 # Assign a single cluster when distance to multiple clusters is same
    #                 dist = F.one_hot(curr_nn_index, self.num_clusters).type(torch.float32)
    #                 curr_centers = self.update_centers_(feat[i*chunk:(i+1)*chunk, :], dist, curr_nn_index, avg=False)
    #                 counts += dist.detach().sum(0) + 1e-6
    #                 centers += curr_centers
    #                 if self.nn_index == None:
    #                     self.nn_index = curr_nn_index
    #                 else:
    #                     self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
    #                 i += 1
    #                 if i*chunk > feat.shape[0]:
    #                     break

    #         self.centers = centers / counts.unsqueeze(-1)
    #         # Reinitialize to 0
    #         centers[centers != 0] = 0.
    #         counts[counts > 0.1] = 0.

    #     if chunk:
    #         self.nn_index = None
    #         i = 0
    #         # chunk = 100000
    #         while True:
    #             dist = self.get_dist(feat_scaled[i * chunk:(i + 1) * chunk, :], self.centers)
    #             curr_nn_index = torch.argmin(dist, dim=-1)
    #             if self.nn_index == None:
    #                 self.nn_index = curr_nn_index
    #             else:
    #                 self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
    #             i += 1
    #             if i * chunk > feat.shape[0]:
    #                 break
    #     self.equalize_cluster_size()


    def rescale(self, feat, scale=None):
        """Scale the feature to be in the range [-1, 1] by dividing by its max value.

        """
        if scale is None:
            return feat / (abs(feat).max(dim=0)[0] + 1e-8)
        else:
            return feat / (scale + 1e-8)

    def forward_pos(self, gaussian, assign=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._xyz.shape[1]
        if assign:
            # print("Dimension of xyz:", gaussian._xyz.shape)
            self.cluster_assign(gaussian._xyz)
        else:
            # print("Dimension of xyz:", gaussian._xyz.shape)
            self.update_centers(gaussian._xyz)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._xyz_q = gaussian._xyz - gaussian._xyz.detach() + sampled_centers

    def forward_dc(self, gaussian, assign=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._features_dc.shape[1] * gaussian._features_dc.shape[2]
        if assign:
            # print("Dimension of features_dc:", gaussian._features_dc.shape)
            self.cluster_assign(gaussian._features_dc)
        else:
            # print("Dimension of features_dc:", gaussian._features_dc.shape)
            self.update_centers(gaussian._features_dc)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_dc_q = gaussian._features_dc - gaussian._features_dc.detach() + sampled_centers.reshape(-1, 1, 3)

    def forward_frest(self, gaussian, assign=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._features_rest.shape[1] * gaussian._features_rest.shape[2]
        if assign:
            # print("Dimension of features_rest:", gaussian._features_rest.shape)
            self.cluster_assign(gaussian._features_rest)
        else:
            # print("Dimension of features_rest:", gaussian._features_rest.shape)
            self.update_centers(gaussian._features_rest)
        deg = gaussian._features_rest.shape[1]
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_rest_q = gaussian._features_rest - gaussian._features_rest.detach() + sampled_centers.reshape(-1, deg, 3)

    def forward_scale(self, gaussian, assign=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._scaling.shape[1]
        if assign:
            # print("Dimension of scaling:", gaussian._scaling.shape)
            self.cluster_assign(gaussian._scaling)
        else:
            # print("Dimension of scaling:", gaussian._scaling.shape)
            self.update_centers(gaussian._scaling)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._scaling_q = gaussian._scaling - gaussian._scaling.detach() + sampled_centers

    def forward_rot(self, gaussian, assign=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._rotation.shape[1]
        if assign:
            # print("Dimension of rotation:", gaussian._rotation.shape)
            self.cluster_assign(gaussian._rotation)
        else:
            # print("Dimension of rotation:", gaussian._rotation.shape)
            self.update_centers(gaussian._rotation)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._rotation_q = gaussian._rotation - gaussian._rotation.detach() + sampled_centers

    def forward_scale_rot(self, gaussian, assign=False):
        """Combine both scaling and rotation for a single k-Means"""
        if self.vec_dim == 0:
            self.vec_dim = gaussian._rotation.shape[1] + gaussian._scaling.shape[1]
        feat_scaled = torch.cat([self.rescale(gaussian._scaling), self.rescale(gaussian._rotation)], 1)
        feat = torch.cat([gaussian._scaling, gaussian._rotation], 1)
        if assign:
            self.cluster_assign(feat, feat_scaled)
        else:
            self.update_centers(feat)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._scaling_q = gaussian._scaling - gaussian._scaling.detach() + sampled_centers[:, :3]
        gaussian._rotation_q = gaussian._rotation - gaussian._rotation.detach() + sampled_centers[:, 3:]

    def forward_dcfrest(self, gaussian, assign=False):
        """Combine both features_dc and rest for a single k-Means"""
        if self.vec_dim == 0:
            self.vec_dim = (gaussian._features_rest.shape[1] * gaussian._features_rest.shape[2] +
                            gaussian._features_dc.shape[1] * gaussian._features_dc.shape[2])
        if assign:
            self.cluster_assign(torch.cat([gaussian._features_dc, gaussian._features_rest], 1))
        else:
            self.update_centers(torch.cat([gaussian._features_dc, gaussian._features_rest], 1))
        deg = gaussian._features_rest.shape[1]
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_dc_q = gaussian._features_dc - gaussian._features_dc.detach() + sampled_centers[:, :3].reshape(-1, 1, 3)
        gaussian._features_rest_q = gaussian._features_rest - gaussian._features_rest.detach() + sampled_centers[:, 3:].reshape(-1, deg, 3)

    def replace_with_centers(self, gaussian):
        deg = gaussian._features_rest.shape[1]
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_rest = gaussian._features_rest - gaussian._features_rest.detach() + sampled_centers.reshape(-1, deg, 3)

