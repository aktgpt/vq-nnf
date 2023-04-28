import time

import numpy as np
import torch
from torch.nn.functional import cosine_similarity


def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    if num_samples < num_clusters:
        indices = np.random.choice(num_samples, num_clusters, replace=True)
    else:
        indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(X, num_clusters, n_iter=200, tol=1e-6):
    # initialize

    initial_state = initialize(X, num_clusters)
    iteration = 0

    torch.cuda.synchronize()
    t1 = time.time()
    while iteration < n_iter:
        dis = pairwise_l2(X, initial_state)  # pairwise_cosine
        # print(dis.min(), dis.max())
        choice_cluster = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            # selected = torch.nonzero(choice_cluster == index).squeeze()  # .to(device)
            selected = torch.where((choice_cluster == index) == True)[0].squeeze()
            selected = torch.index_select(X, 0, selected)

            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1))
        )

        # print(center_shift.item())
        if torch.isnan(center_shift):
            print("ReInitializing...")
            print(torch.sum(torch.isnan(dis)).item())
            print(torch.sum(torch.isnan(initial_state)).item())
            initial_state = initialize(X, num_clusters)
            iteration = 0

        # increment iteration
        iteration = iteration + 1

        if center_shift ** 2 < tol:
            break
    torch.cuda.synchronize()
    t2 = time.time()
    # print(
    #     f"KMeans Clustering Time: {time.time()-t1} sec. Stopped after {iteration} Iterations with shift {center_shift:.6f}"
    # )
    # print(iteration)
    return initial_state, choice_cluster, dis, t2 - t1


def pairwise_l2(data1, data2):
    return torch.cdist(data1, data2)


def pairwise_cosine(data1, data2):
    data1 = data1 / torch.norm(data1, dim=1, keepdim=True)
    data2 = data2 / torch.norm(data2, dim=1, keepdim=True)
    similarity_mat = torch.matmul(data1, data2.T).clip(0, 1)
    cosine_dis = 1 - similarity_mat
    return cosine_dis


def SSE(points):
    """
    Calculates the sum of squared errors for the given list of data points.
    """
    centroid = torch.mean(points, 0)
    errors = torch.sum((points - centroid) ** 2, 1)
    # errors = np.linalg.norm(points - centroid, ord=2, axis=1)
    return torch.sum(errors)


def bisecting_kmeans(X, num_clusters=2, n_iter=100):
    """
    Clusters the list of points into `k` clusters using bisecting k-means
    clustering algorithm. Internally, it uses the standard k-means with k=2 in
    each iteration.
    """
    clusters = [X]
    cluster_indices = [torch.arange(len(X))]
    while len(clusters) < num_clusters:
        sses = [SSE(c).item() for c in clusters]
        max_sse_i = np.argmax(sses)
        cluster = clusters.pop(max_sse_i)
        cluster_index = cluster_indices.pop(max_sse_i)
        cluster_centers, choice_cluster, dis_vals = kmeans(cluster, 2, n_iter=n_iter)
        two_clusters = [cluster[choice_cluster == 0], cluster[choice_cluster == 1]]
        two_clusters_index = [
            cluster_index[choice_cluster == 0],
            cluster_index[choice_cluster == 1],
        ]
        clusters.extend(two_clusters)
        cluster_indices.extend(two_clusters_index)

    all_cluster_centers = []
    all_cluster_labels = torch.zeros(len(X), dtype=torch.long, device=X.device)
    for i, (cluster, cluster_index) in enumerate(zip(clusters, cluster_indices)):
        all_cluster_centers.append(torch.mean(cluster, dim=0).unsqueeze(0))
        all_cluster_labels[cluster_index] = i
    all_cluster_centers = torch.cat(all_cluster_centers, dim=0)
    all_dis_vals = pairwise_l2(X, all_cluster_centers)
    return all_cluster_centers, all_cluster_labels, all_dis_vals

