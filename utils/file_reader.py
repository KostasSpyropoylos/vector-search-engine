import numpy as np

class FileReader:
    @staticmethod
    def read_fvecs(file_path, num_vectors=None):
        if num_vectors is not None:
            with open(file_path, 'rb') as f:
                dim_arr = np.fromfile(f, dtype='int32', count=1)
                if len(dim_arr) == 0:
                    return np.empty((0, 0), dtype='float32')
                dim = dim_arr[0]
                f.seek(0)
                arr = np.fromfile(f, dtype='int32', count=num_vectors * (dim + 1))
        else:
            arr = np.fromfile(file_path, dtype='int32')
            if len(arr) == 0:
                return np.empty((0, 0), dtype='float32')
            dim = arr[0]
        
        num_complete_vectors = len(arr) // (dim + 1)
        arr = arr[:num_complete_vectors * (dim + 1)]
        vectors = arr.reshape(-1, dim + 1)[:, 1:].copy()
        return vectors.view('float32')


    @staticmethod
    def write_fvecs_to_txt(file_path, vectors):
        import os
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for vector in vectors:
                f.write(", ".join(f"{val:.4f}" for val in vector) + "\n")
        print(f"Vectors successfully written to {file_path}")