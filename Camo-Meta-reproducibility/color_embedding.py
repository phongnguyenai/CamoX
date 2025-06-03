import numpy as np
import torch
from PIL import Image

# class KMeansCUDA:
#     def __init__(self, n_clusters, min_diff=1e-4, max_iter=1000, device='cuda'):
#         self.n_clusters = n_clusters
#         self.min_diff = min_diff
#         self.max_iter = max_iter
#         self.device = device

#     def fit(self, points):
#         # Convert points to PyTorch tensor and move to the specified device (GPU)
#         # points = torch.tensor(points, dtype=torch.float32).to(self.device)

#         # Randomly initialize cluster centers
#         initial_indices = torch.randperm(points.shape[0])[:self.n_clusters]
#         centers = points[initial_indices]

#         for _ in range(self.max_iter):
#             # Calculate distances between points and cluster centers
#             distances = torch.cdist(points, centers)

#             # Assign points to the nearest cluster center
#             cluster_assignments = torch.argmin(distances, dim=1)

#             # Calculate new cluster centers
#             new_centers = torch.stack([points[cluster_assignments == k].mean(dim=0) for k in range(self.n_clusters)])

#             # Calculate the maximum change in cluster centers
#             center_shift = torch.norm(new_centers - centers, dim=1).max()

#             # Update cluster centers
#             centers = new_centers

#             # Check for convergence
#             if center_shift < self.min_diff:
#                 break

#         return centers.cpu().detach()  # Move centers back to CPU

class KMeansCUDA:
    def __init__(self, n_clusters, min_diff=1e-4, max_iter=100, device='cuda', epsilon=1e-8):
        self.n_clusters = n_clusters
        self.min_diff = min_diff
        self.max_iter = max_iter
        self.device = device
        self.epsilon = epsilon

    def _initialize_centers(self, points):
        # k-means++ initialization
        centers = [points[torch.randint(points.shape[0], (1,)).item()]]
        for _ in range(1, self.n_clusters):
            distances = torch.cdist(points, torch.stack(centers)).min(dim=1)[0]
            probabilities = distances / (distances.sum() + self.epsilon)
            # Handle large probabilities by downsampling
            sample_size = min(2**24 - 1, probabilities.size(0))
            sampled_indices = torch.multinomial(probabilities, sample_size, replacement=True)
            sampled_probabilities = probabilities[sampled_indices]
            
            new_center_idx = sampled_indices[torch.multinomial(sampled_probabilities, 1).item()]
            centers.append(points[new_center_idx])
        return torch.stack(centers)

    def _reinitialize_empty_clusters(self, points, cluster_assignments):
        unique_clusters = cluster_assignments.unique()
        for cluster_id in range(self.n_clusters):
            if cluster_id not in unique_clusters:
                new_center = points[torch.randint(points.shape[0], (1,)).item()]
                cluster_assignments[torch.argmin(torch.cdist(points, new_center.unsqueeze(0)))].fill_(cluster_id)

    def fit(self, points):
        # Ensure points are a PyTorch tensor and move to the specified device
        # points = torch.tensor(points, dtype=torch.float32).to(self.device)

        # Initialize cluster centers
        centers = self._initialize_centers(points)

        for _ in range(self.max_iter):
            # Calculate distances between points and cluster centers
            distances = torch.cdist(points, centers)

            # Assign points to the nearest cluster center
            cluster_assignments = torch.argmin(distances, dim=1)

            # Reinitialize empty clusters
            self._reinitialize_empty_clusters(points, cluster_assignments)

            # Calculate new cluster centers
            new_centers = torch.stack([points[cluster_assignments == k].mean(dim=0) for k in range(self.n_clusters)])

            # Calculate the maximum change in cluster centers
            center_shift = torch.norm(new_centers - centers, dim=1).max()

            # Update cluster centers
            centers = new_centers

            # Check for convergence
            if center_shift < self.min_diff:
                break

        # Count the number of points assigned to each cluster
        cluster_counts = torch.bincount(cluster_assignments, minlength=self.n_clusters)
        # Sort indices based on sums
        sorted_indices = torch.argsort(cluster_counts, descending=True)
        
        # Sort the tensor based on sorted indices
        sorted_centers = centers[sorted_indices]

        return sorted_centers.cpu().detach()