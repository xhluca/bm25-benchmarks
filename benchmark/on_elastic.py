import json
import os
from pathlib import Path
import time

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import numpy as np
from tqdm.auto import tqdm
from beir.retrieval.search.lexical import BM25Search

from utils.benchmark import get_max_memory_usage, Timer
from utils.beir import merge_cqa_dupstack, clean_results_keys

def compute_top_k_from_scores(
    scores, corpus=None, k=10, sorting=False, with_scores=False
):
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)

    # top_n = np.argsort(scores)[::-1][:n]
    top_n = np.argpartition(scores, -k)[-k:]
    if sorting:
        # sort in descending order
        top_n = top_n[np.argsort(scores[top_n])][::-1]

    if corpus is None:
        results = top_n
    else:
        results = [corpus[i] for i in top_n]

    if with_scores:
        top_scores = scores[top_n]
        return results, top_scores
    else:
        return results


def main(
    dataset,
    n_threads=1,
    top_k=1000,
    save_dir="datasets",
    result_dir="results",
    hostname = "localhost",
    k1=1.2,
    b=0.75,
):
    #### Download dataset and unzip the dataset
    base_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
    data_path = beir.util.download_and_unzip(base_url.format(dataset), save_dir)

    if dataset == "msmarco":
        split = "dev"
    else:
        split = "test"

    if dataset == "cqadupstack":
        merge_cqa_dupstack(data_path)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    num_docs = len(corpus)
    num_queries = len(queries)

    print("=" * 50)
    print("Dataset: ", dataset)
    print(f"Corpus Size: {num_docs:,}")
    print(f"Queries Size: {num_queries:,}")
    timer = Timer("[Elastic-BM25]")
    
    model = BM25Search(
        index_name=dataset, hostname=hostname, language="english", number_of_shards=1, initialize=False
    )
    es_bm25_settings = {
        "settings": {
            "index": {
                "similarity": {
                    "default": {
                        "type": "BM25",
                        "k1": k1,
                        "b": b,
                    }
                }
            },
            "analysis": {
                "analyzer": {
                    "custom_analyzer": {
                        "type": "standard",
                        "max_token_length": 1_000_000,
                        "stopwords": "_english_",
                        "filter": [ "lowercase", "custom_snowball"]
                    }
                },
                "filter": {
                    "custom_snowball": {
                        "type": "snowball",
                        "language": "English"
                    }
                }
            }
        }
    }
    model.initialise()

    t = timer.start("Index")
    model.index(corpus)
    timer.stop(t, show=True, n_total=num_docs)

    model.es.es.indices.close(index=dataset)
    model.es.es.indices.put_settings(index=dataset, body=es_bm25_settings)
    model.es.es.indices.open(index=dataset)
    
    t_query = timer.start("Query")
    results = model.search(corpus=corpus, queries=queries, top_k=top_k)    
    timer.stop(t_query, show=True, n_total=num_queries)

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, [1, 10, 100, 1000]
    )

    max_mem_gb = get_max_memory_usage("GB")

    print("=" * 50)
    print(f"Max Memory Usage: {max_mem_gb:.4f} GB")
    print("-" * 50)
    print(ndcg)
    print(recall)
    print("=" * 50)

    # Save everything to json
    save_dict = {
        "model": "elastic-bm25",
        "dataset": dataset,
        "stemmer": "snowball",
        "tokenizer": "elastic",
        "k1": k1,
        "b": b,
        "method": "bm25",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_threads": n_threads,
        "top_k": top_k,
        "max_mem_gb": max_mem_gb,
        "stats": {
            "num_docs": num_docs,
            "num_queries": num_queries,
        },
        "timing": timer.to_dict(underscore=True, lowercase=True),
        "scores": {
            "ndcg": clean_results_keys(ndcg),
            "map": clean_results_keys(_map),
            "recall": clean_results_keys(recall),
            "precision": clean_results_keys(precision),
        },
    }

    result_dir = Path(result_dir) / save_dict["model"]
    result_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(result_dir) / f"{dataset}-{os.urandom(8).hex()}.json"
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark rank-bm25 on a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="fiqa",
        help="Dataset to benchmark on.",
    )

    parser.add_argument(
        "-t",
        "--n_threads",
        type=int,
        default=1,
        help="Number of threads to run in parallel.",
    )

    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of runs to repeat main."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="Number of top-k documents to retrieve.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="datasets",
        help="Directory to save datasets.",
    )

    parser.add_argument(
        "--hostname",
        type=str,
        default="localhost",
        help="Hostname of the ElasticSearch server.",
    )

    parser.add_argument(
        "--k1",
        type=float,
        default=1.2,
        help="BM25 k1 parameter.",
    )

    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter.",
    )

    kwargs = vars(parser.parse_args())
    profile = kwargs.pop("profile")
    num_runs = kwargs.pop("num_runs")

    if profile:
        import cProfile
        import pstats

        if num_runs > 1:
            raise ValueError("Cannot profile with multiple runs.")

        cProfile.run("main(**kwargs)", filename="rankbm25.prof")
        p = pstats.Stats("rankbm25.prof")
        p.sort_stats("time").print_stats(50)
    else:
        for _ in range(num_runs):
            main(**kwargs)
