from sklearn.cluster import DBSCAN


def load_DBSCAN(eps=1.2, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan
