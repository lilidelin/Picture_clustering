from sklearn.cluster import DBSCAN


def load_DBSCAN(eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan
