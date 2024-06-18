import argparse
import json
import resource

import Stemmer
import numpy as np
import beir.util
from beir.datasets.data_loader import GenericDataLoader

import bm25s
from bm25s.utils.benchmark import get_max_memory_usage, Timer
from bm25s.utils.beir import BASE_URL

def main(save_dir, data_dir, dataset):
    data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), data_dir)

    loader = GenericDataLoader(data_folder=data_path)
    loader._load_queries()
    queries = loader.queries

    timer = Timer("[BM25S]")
    # now, load the index
    model = bm25s.BM25.load(f"{save_dir}/{dataset}", mmap=False, load_corpus=True)
    
    qids, queries_lst = [], []
    for key, val in queries.items():
        qids.append(key)
        queries_lst.append(val)
    
    queries_lst = queries_lst[:1000]
    stemmer = Stemmer.Stemmer("english")
    queries_tokenized = bm25s.tokenize(queries_lst, stopwords="en", stemmer=stemmer)

    t = timer.start("Query")
    res = model.retrieve(
        queries_tokenized,
        k=20,
        backend_selection="jax",
    )
    timer.stop(t, show=True, n_total=len(queries_lst))

    max_mem_gb = get_max_memory_usage("GB")
    print(f"Max Memory Usage: {max_mem_gb:.2f} GB")

def parse_args():
    parser = argparse.ArgumentParser(description="BM25s Benchmark")
    parser.add_argument("--save_dir", type=str, default="bm25s_indices", help="Directory where the index is saved")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Directory where we save the dataset")
    parser.add_argument("-d", "--dataset", type=str, default="quora", help="Dataset to use for benchmarking")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))