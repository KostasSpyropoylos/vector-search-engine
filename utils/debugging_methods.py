from utils.file_reader import FileReader

def available_debugging_methods(dataset,queries,engine):
    while True:
        print("\n------------------------------------ Debugging Methods ------------------------------------ \n")
        print("1. Write dataset to txt file")
        print("2. Write queries to txt file")
        print("3. Write inverted index to txt file")
        print("4. Write clusters visualization to png file")
        print("5. Write KNN results to txt file")
        print("6. Continue to evaluation")
        print("\n-------------------------------------------------------------------------------------------")
        choice = input("Enter your choice: ")
        if choice == "6":
            break
        if choice == "1":
            FileReader.write_fvecs_to_txt('output/dataset.txt', dataset)
            print("\n------------------------------------\nDataset written to output/dataset.txt\n------------------------------------")
        elif choice == "2":
            FileReader.write_fvecs_to_txt('output/queries.txt', queries)
            print("\n------------------------------------\nQueries written to output/queries.txt\n------------------------------------")
        elif choice == "3":
            engine.write_index_to_file()
            print("\n------------------------------------\nInverted index written to output/inverted_index.txt\n------------------------------------")
        elif choice == "4":
            engine.visualize_clusters(queries)
            print("\n------------------------------------\nClusters visualization written to output/clusters_visualization.png\n------------------------------------")
        elif choice == "5":
            engine.write_knn_results(queries)
            print("\n------------------------------------\nKNN results written to output/knn_results.txt\n------------------------------------")
