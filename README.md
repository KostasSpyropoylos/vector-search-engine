# Vector Search Engine for High-Dimensional Vectors

An efficient Python-based implementation of a **Vector Search Engine** utilizing exact and approximate K-Nearest Neighbors (K-NN) search algorithms. Developed as part of the **Information Retrieval (Ανάκτηση Πληροφοριών)** coursework.

This engine implements the **Inverted File Index (IVF-flat)** architecture using $K$-Means clustering to achieve sub-linear query processing times on high-dimensional datasets. It is benchmarked against the standard **SIFT1M** dataset containing 1,000,000 128-dimensional vectors.

---

## Features

- **Exhaustive Exact K-NN Search:** Computes exact squared Euclidean distances to all dataset vectors to guarantee 100% accuracy.
- **Inverted File Index (IVF-Flat) Approximate Search:** Partitions the vector space into $P$ clusters using $K$-Means and limits candidate scanning to the $M$ most promising clusters.
- **Fast Binary Reading:** Handled by a specialized parser for the `.fvecs` format to load large vector datasets quickly.
- **Comprehensive Evaluation Pipeline:** Compares performance metrics such as **Queries Per Second (QPS)**, **Execution Time (seconds)**, and **Recall@K** to visualize the accuracy-speed tradeoff.

---

## Project Structure

```text
├── dataset/                     # Directory for dataset files (ignored by Git)
│   ├── sift_base.fvecs          # 1,000,000 base vectors (128-d)
│   ├── sift_query.fvecs         # 10,000 query vectors (128-d)
│   ├── sift_learn.fvecs         # 100,000 learning vectors (128-d)
│   └── sift_groundtruth.ivecs   # Ground truth nearest neighbors
├── models/
│   ├── __init__.py
│   └── search_engine.py         # VectorSearchEngine class (exact & approx search)
├── utils/
│   ├── __init__.py
│   ├── file_reader.py           # fvecs file parsing helper
│   └── evalution_util.py        # Evaluation pipeline (Recall, QPS, Latency)
├── main.py                      # Main entrypoint script
├── .gitignore                   # Standard Python Git ignore rules
└── README.md                    # Project documentation
```

---

## Dataset Setup

The engine is benchmarked on the **SIFT1M** dataset.

1. **Download the SIFT dataset:**
   - URL: [SIFT1M Corpus (ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz)](ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz)
2. **Extract files:**
   - Extract the contents into the `dataset/` directory inside the repository.
   - Verify that `sift_base.fvecs` and `sift_query.fvecs` are present inside `dataset/`.

---

## Installation & Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/KostasSpyropoylos/vector-search-engine.git
   cd vector-search-engine
   ```

2. **Set up a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

To build the index, run exact vs approximate search, and output performance metrics, execute the following command:

```bash
python3 main.py
```
