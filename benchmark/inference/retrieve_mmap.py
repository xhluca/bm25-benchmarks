import argparse
import json
import resource

import Stemmer
import numpy as np

import bm25s
from bm25s.utils.benchmark import get_max_memory_usage, Timer


def main(save_dir, dataset):
    timer = Timer("[BM25S]")
    # now, load the index
    t = timer.start("Loading index (mmap)")
    model = bm25s.BM25.load(f"{save_dir}/{dataset}", mmap=True, load_corpus=True)
    timer.stop(t, show=True)


    query = [
        "What is the incubation period of COVID-19?",
        "Can COVID-19 be transmitted through food?",
        "If I have COVID-19, can I breastfeed my child?",
        "When should I get tested for COVID-19?",
        "Would COVID-19 be transmitted through blood transfusion?",
        "How do you properly wear a mask?",
    ]
    stemmer = Stemmer.Stemmer("english")
    query_tokenized = bm25s.tokenize(query, stopwords="en", stemmer=stemmer)
    t = timer.start("Retrieving documents (mmap)")
    res = model.retrieve(query_tokenized, k=5)
    timer.stop(t, show=True)

    max_mem_gb = get_max_memory_usage("GB")

    print(f"Max Memory Usage (mmap): {max_mem_gb:.2f} GB")
    print()

    # now, do the same thing with no mmap, then compare the memory usage and results
    t = timer.start("Loading index (in-memory)")
    model_no_mmap = bm25s.BM25.load(f"{save_dir}/{dataset}", mmap=False, load_corpus=True)
    timer.stop(t, show=True)

    t = timer.start("Retrieving documents (in-memory)")
    res_no_mmap = model_no_mmap.retrieve(query_tokenized, k=5)
    timer.stop(t, show=True)

    max_mem_gb2 = get_max_memory_usage("GB")
    print(f"Max Memory Usage (in-memory): {max_mem_gb2:.2f} GB")

    d1 = res.documents[0].tolist()
    d2 = res_no_mmap.documents[0].tolist()

    assert d1 == d2, "Results are not the same!"

    print("Results are the same!")

def parse_args():
    parser = argparse.ArgumentParser(description="BM25s Benchmark")
    parser.add_argument("--save_dir", type=str, default="bm25s_indices", help="Directory where the index is saved")
    parser.add_argument("-d", "--dataset", type=str, default="quora", help="Dataset to use for benchmarking")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))