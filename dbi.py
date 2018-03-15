import numpy as np


def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def avg_dist_intra_cluster(cluster):
    """Computes average intra-cluster distance which is neccessary to computer
    DBI.

    Args:
        cluster: An cluster = {x1, x2,..., xk}. xi is a row.

    Returns:
        A float number of average intra-cluster distance.
    """
    sigma = 0
    size = cluster.shape[0]
    for i in range(size):
        for j in range(size):
            if i < j:
                sigma += euclidean_dist(cluster[i], cluster[j])
    return 2 * sigma / (size * (size - 1))


def center_point(cluster):
    center = np.sum(cluster, axis=0)
    return center


def dist_inter_cluster(cluster1, cluster2):
    center1 = center_point(cluster1)
    center2 = center_point(cluster2)
    distance = euclidean_dist(center1, center2)
    return distance
    

