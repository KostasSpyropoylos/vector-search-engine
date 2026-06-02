import os
import numpy as np
from utils.file_reader import FileReader



if __name__ == "__main__":
    dataset=FileReader.read_fvecs('sift/sift_base.fvecs')
    queries=FileReader.read_fvecs('sift/sift_query.fvecs')