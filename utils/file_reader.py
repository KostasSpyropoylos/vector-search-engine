import numpy as np

class FileReader:
    def read_fvecs(file_path):
        arr = np.fromfile(file_path, dtype='int32')
        dim = arr[0]
        vectors = arr.reshape(-1, dim + 1)[:, 1:].copy()
        return vectors.view('float32')
