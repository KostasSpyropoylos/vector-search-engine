import numpy as np
from scipy.cluster.vq import kmeans2
import time

class VectorSearchEngine:
    def __init__(self, data):
        """
        Initializes the search engine with the base dataset.
        data: numpy array of shape (N, D) where N is number of vectors, D is dimensions.
        """
        self.data = data
        self.N, self.D = data.shape
        self.centroids = None
        self.inverted_index = {}
