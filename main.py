import os
import numpy as np
from models.search_engine import VectorSearchEngine
from utils.evalution_util import evaluate_search
from utils.file_reader import FileReader
from utils.debugging_methods import available_debugging_methods

if __name__ == "__main__":
    dataset=FileReader.read_fvecs('dataset/sift_base.fvecs', 10)
    queries=FileReader.read_fvecs('dataset/sift_query.fvecs', 10)
    engine = VectorSearchEngine(dataset)
    
    P = int(np.sqrt(dataset.shape[0])) 
    engine.build_inverted_index(num_clusters_p=P)
    
    M = max(1, P // 10)

    available_debugging_methods(dataset,queries,engine)
    evaluate_search(engine, queries, k=100, m_groups=M)