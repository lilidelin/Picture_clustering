from sklearn.cluster import KMeans


def load_KMeans(num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    return kmeans
