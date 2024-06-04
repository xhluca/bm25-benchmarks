import json
import os
from pathlib import Path
import random
import time

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import numpy as np
from tqdm.auto import tqdm
import Stemmer
import rank_bm25

import utils
from utils.benchmark import get_max_memory_usage, Timer
from utils.beir import (
    BASE_URL,
    clean_results_keys,
    merge_cqa_dupstack,
    postprocess_results_for_eval,
)


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
    method="rank",
    n_threads=1,
    top_k=1000,
    save_dir="datasets",
    result_dir="results",
    samples=0,
    verbose=False,
):
    #### Download dataset and unzip the dataset
    data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), save_dir)

    if dataset == "cqadupstack":
        merge_cqa_dupstack(data_path)

    if dataset == "msmarco":
        split = "dev"
    else:
        split = "test"

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    if samples > 0:
        random.seed(42)
        query_keys = list(queries.keys())
        query_keys = random.sample(query_keys, samples)
        queries = {k: queries[k] for k in query_keys}

    num_docs = len(corpus)
    corpus_ids, corpus_lst = [], []
    for key, val in corpus.items():
        corpus_ids.append(key)
        corpus_lst.append(val["title"] + " " + val["text"])

    del corpus

    qids, queries_lst = [], []
    for key, val in queries.items():
        qids.append(key)
        queries_lst.append(val)

    print("=" * 50)
    print("Dataset: ", dataset)
    print(f"Corpus Size: {num_docs:,}")
    print(f"Queries Size: {len(queries_lst):,}")

    stemmer = Stemmer.Stemmer("english")
    timer = Timer("[Rank-BM25]")
    t = timer.start("Tokenize Corpus")
    tokenized_corpus = utils.tokenize(
        corpus_lst,
        stopwords="en",
        stemmer=stemmer,
        leave=False,
    )
    timer.stop(t, show=True, n_total=num_docs)

    del corpus_lst

    t = timer.start("Tokenize Queries")
    queries_tokenized = utils.tokenize(queries_lst, stopwords="en", stemmer=stemmer)
    timer.stop(t, show=True, n_total=len(queries_lst))

    num_tokens = sum(len(doc) for doc in tokenized_corpus)
    print(f"Number of Tokens: {num_tokens:,}")
    print(f"Number of Tokens / Doc: {num_tokens / num_docs:.2f}")
    print("-" * 50)

    t = timer.start("Index")
    if method == "rank":
        model = rank_bm25.BM25Okapi(
            corpus=tokenized_corpus, epsilon=0.0, k1=1.5, b=0.75
        )
    elif method == "bm25l":
        model = rank_bm25.BM25L(corpus=tokenized_corpus, k1=1.5, b=0.75, delta=0.5)
    elif method == "bm25+":
        model = rank_bm25.BM25Plus(corpus=tokenized_corpus, k1=1.5, b=0.75, delta=0.5)
    else:
        raise ValueError(f"Unknown method: {method}")

    timer.stop(t, show=True, n_total=num_docs)

    results = []
    scores = []

    t_score = timer.start("Score")
    timer.pause("Score")
    t_query = timer.start("Query")
    for q in tqdm(
        queries_tokenized, desc="Rank-BM25 Scoring", leave=False, disable=not verbose
    ):
        timer.resume(t_score)
        raw_scores = model.get_scores(q)
        timer.pause(t_score)
        result, score = compute_top_k_from_scores(
            raw_scores, corpus=corpus_ids, k=top_k, with_scores=True
        )
        results.append(result)
        scores.append(score)

    queried_results = np.array(results)
    queried_scores = np.array(scores)

    timer.stop(t_score, show=True, n_total=len(queries_lst))
    timer.stop(t_query, show=True, n_total=len(queries_lst))

    results_dict = postprocess_results_for_eval(queried_results, queried_scores, qids)
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results_dict, [1, 10, 100, 1000]
    )

    max_mem_gb = get_max_memory_usage("GB")

    print("-" * 50)
    print(f"Max Memory Usage: {max_mem_gb:.4f} GB")
    print(ndcg)
    print(recall)
    print("=" * 50)

    # Save everything to json
    save_dict = {
        "model": "rank-bm25",
        "dataset": dataset,
        "stemmer": "snowball",
        "tokenizer": "skl",
        "method": method,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_threads": n_threads,
        "samples": samples,
        "top_k": top_k,
        "max_mem_gb": max_mem_gb,
        "stats": {
            "num_docs": num_docs,
            "num_queries": len(queries_lst),
            "num_tokens": num_tokens,
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
        "--samples",
        type=int,
        default=0,
        help="Number of samples to use from the dataset. If 0, use all samples.",
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
        "--method",
        type=str,
        default="rank",
        choices=["rank", "bm25l", "bm25+"],
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
