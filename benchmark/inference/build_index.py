import argparse
import beir.util
from beir.datasets.data_loader import GenericDataLoader
import Stemmer

import bm25s
from bm25s.utils.beir import BASE_URL
from bm25s.utils.benchmark import get_max_memory_usage, Timer


def main(save_dir, index_dir, dataset):
    data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), save_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    num_docs = len(corpus)

    corpus_ids, corpus_lst, corpus_records = [], [], []
    for key, val in corpus.items():
        corpus_ids.append(key)
        corpus_lst.append(val["title"] + " " + val["text"])
        corpus_records.append({'id': key, 'title': val["title"], 'text': val["text"]})

    stemmer = Stemmer.Stemmer("english")
    corpus_tokenized = bm25s.tokenize(corpus_lst, stopwords="en", stemmer=stemmer)

    model = bm25s.BM25(corpus=corpus_records)
    model.index(corpus_tokenized)
    # save the model
    model.save(f"{index_dir}/{dataset}")

    max_mem_gb = get_max_memory_usage("GB")

    print(f"Max Memory Usage: {max_mem_gb:.2f} GB")


def parse_args():
    parser = argparse.ArgumentParser(description="BM25s Benchmark")
    parser.add_argument("--save_dir", type=str, default="datasets", help="Directory where we save the dataset")
    parser.add_argument("--index_dir", type=str, default="bm25s_indices", help="Directory where the index is saved")
    parser.add_argument("-d", "--dataset", type=str, default="quora", help="Dataset to use for benchmarking")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))