import time

def evaluate_search(engine, queries, k=10, m_groups=5):
    """
    Evaluates exact vs approximate algorithms, computing QPS and Recall.
    """
    num_queries = len(queries)
    queries = engine.pca.transform(queries)
    # Run Exact K-NN
    exact_results = []
    start_time = time.time()
    for q in queries:
        indices, _ = engine.exact_knn(q, k)
        exact_results.append(set(indices))
    exact_time = time.time() - start_time
    exact_qps = num_queries / exact_time
    
    # Run Approximate K-NN
    approx_results = []
    start_time = time.time()
    for q in queries:
        indices, _ = engine.approximate_knn(q, k, m_groups)
        approx_results.append(set(indices))
    approx_time = time.time() - start_time
    approx_qps = num_queries / approx_time
    
    # Calculate Recall
    total_recall = 0
    for exact_set, approx_set in zip(exact_results, approx_results):
        intersection = len(exact_set.intersection(approx_set))
        total_recall += intersection / k
    average_recall = total_recall / num_queries
    
    print("\n--- Evaluation Results ---")
    print(f"Queries: {num_queries}, K: {k}")
    print(f"Exact Algorithm       - Time: {exact_time:.4f}s | QPS: {exact_qps:.2f}")
    print(f"Approximate Algorithm - Time: {approx_time:.4f}s | QPS: {approx_qps:.2f} | M Groups: {m_groups}")
    print(f"Average Recall        : {average_recall * 100:.2f}%")

