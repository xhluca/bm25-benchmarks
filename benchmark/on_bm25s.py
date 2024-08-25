import json
import os
from pathlib import Path
import time

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import numpy as np
from tqdm.auto import tqdm
import Stemmer
from numba import njit

import bm25s
from bm25s.utils.benchmark import get_max_memory_usage, Timer
from bm25s.utils.beir import (
    BASE_URL,
    clean_results_keys,
    merge_cqa_dupstack,
    postprocess_results_for_eval,
)


def main(
    dataset,
    n_threads=1,
    top_k=1000,
    method="lucene",
    save_dir="datasets",
    result_dir="results",
    stopwords="en",
    stemmer_name="snowball",
    k1=1.5,
    b=0.75,
    delta=0.5,
    skip_scoring=False,
    skip_numpy_retrieval=False,
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
    num_docs = len(corpus)

    corpus_ids, corpus_lst = [], []
    for key, val in corpus.items():
        corpus_ids.append(key)
        corpus_lst.append(val["title"] + " " + val["text"])

    corpus_ids = np.array(corpus_ids)
    del corpus

    qids, queries_lst = [], []
    for key, val in queries.items():
        qids.append(key)
        queries_lst.append(val)

    print("=" * 50)
    print("Dataset: ", dataset)
    print(f"Corpus Size: {num_docs:,}")
    print(f"Queries Size: {len(queries_lst):,}")
    print(f"Number of Threads: {n_threads}")

    stemmer_name = None if stemmer_name == "none" else stemmer_name
    stopwords = None if stopwords == "none" else stopwords
    stemmer = Stemmer.Stemmer("english") if stemmer_name == "snowball" else None
    
    timer = Timer("[BM25S]")


    t = timer.start("Tokenize Corpus")
    corpus_tokenized = bm25s.tokenize(
        corpus_lst,
        stopwords=stopwords,
        stemmer=stemmer,
        leave=False,
        return_ids=True,
    )
    timer.stop(t, show=True, n_total=num_docs)

    # t = timer.start("Tokenize Queries with vocab")
    # queries_tokens_ids = bm25s.tokenization._tokenize_with_vocab_exp(
    #     queries_lst,
    #     stopwords=stopwords,
    #     leave=False,
    #     vocab_dict=corpus_tokenized.vocab,
    # )
    # timer.stop(t, show=True, n_total=len(queries_lst))

    t = timer.start("Tokenize Queries")
    queries_tokenized = bm25s.tokenize(
        queries_lst,
        stopwords=stopwords,
        stemmer=stemmer,
        leave=False,
        return_ids=False,
    )
    timer.stop(t, show=True, n_total=len(queries_lst))

    del corpus_lst
    

    num_tokens = sum(len(doc) for doc in corpus_tokenized.ids)
    num_query_tokens = sum(len(q) for q in queries_tokenized)
    num_queries = len(queries_lst)
    print(f"Number of Corpus Tokens: {num_tokens:,}")
    print(f"Number of Tokens / Doc: {num_tokens / num_docs:.2f}")
    print(f"Number of Tokens / Query: {num_query_tokens / num_queries:.2f}")
    print("-" * 50)

    t = timer.start("Index")
    model = bm25s.BM25(method=method, k1=k1, b=b, delta=delta)
    model.index((corpus_tokenized.ids, corpus_tokenized.vocab), leave_progress=False)
    timer.stop(t, show=True, n_total=num_docs)
    
    if not skip_scoring:
        t = timer.start("Score")
        for q in tqdm(queries_tokenized, desc="BM25S Scoring", leave=False):
            model.get_scores(q)
        timer.stop(t, show=True, n_total=len(queries_lst))

        model._compute_relevance_from_scores = bm25s.scoring._compute_relevance_from_scores_legacy
        t = timer.start("Score (legacy)")
        for q in tqdm(queries_tokenized, desc="BM25S Scoring (legacy)", leave=False):
            model.get_scores(q)
        timer.stop(t, show=True, n_total=len(queries_lst))

        # Use njit and warmup
        model._compute_relevance_from_scores = njit(bm25s.scoring._compute_relevance_from_scores_jit_ready)
        model.get_scores(queries_tokenized[0])

        t = timer.start("Score (jit)")
        for q in tqdm(queries_tokenized, desc="BM25S Scoring (jit)", leave=False):
            model.get_scores(q)
        timer.stop(t, show=True, n_total=len(queries_lst))
    
    # Use njit and warmup
    model._compute_relevance_from_scores = njit(bm25s.scoring._compute_relevance_from_scores_jit_ready)
    model.get_scores(queries_tokenized[0])

    ############## BENCHMARKING BEIR HERE ##############

    t = timer.start("Query")
    queried_results, queried_scores = model.retrieve(
        queries_tokenized,
        corpus=corpus_ids,
        k=top_k,
        return_as="tuple",
        n_threads=n_threads,
        backend_selection="jax",
    )
    timer.stop(t, show=True, n_total=len(queries_lst))

    # warmup
    model.retrieve_numba(queries_tokenized[0:2], sorted=True, show_progress=False)
    t = timer.start("Query numba")
    queried_results_nbs, queried_scores_nbs = model.retrieve_numba(
        queries_tokenized,
        corpus=corpus_ids,
        k=top_k,
        return_as="tuple",
        n_threads=n_threads,
        backend_selection="numba",
        sorted=True,
    )
    timer.stop(t, show=True, n_total=len(queries_lst))
    assert np.allclose(queried_scores, queried_scores_nbs, atol=1e-6)

    if not skip_numpy_retrieval:
        t = timer.start("Query numpy")
        queried_results, queried_scores_np = model.retrieve(
            queries_tokenized,
            corpus=corpus_ids,
            k=top_k,
            return_as="tuple",
            n_threads=n_threads,
            backend_selection="numpy",
            sorted=True,
        )
        timer.stop(t, show=True, n_total=len(queries_lst))

        # verify that both results are the same
        assert queried_scores.shape == queried_scores_np.shape
        assert np.allclose(queried_scores, queried_scores_np, atol=1e-6)

    queried_results = queried_results_nbs
    queried_scores = queried_scores_nbs
    
    results_dict = postprocess_results_for_eval(queried_results, queried_scores, qids)
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results_dict, [1, 10, 100, 1000]
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
        "model": "bm25s",
        "dataset": dataset,
        "stemmer": stemmer_name,
        "tokenizer": "skl",
        "method": method,
        "stopwords": stopwords,
        "k1": k1,
        "b": b,
        "delta": delta,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_threads": n_threads,
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

    result_dir = Path(result_dir) / "bm25s"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(result_dir) / f"{dataset}-{os.urandom(8).hex()}.json"
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark bm25s on a dataset.",
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
        "--method",
        type=str,
        default="lucene",
        choices=["lucene", "atire", "robertson", "bm25l", "bm25+"],
        help="Method to use for BM25S.",
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
        "--stopwords",
        type=str,
        default="en",
        choices=["en", "none"],
    )
    parser.add_argument(
        "--stemmer_name",
        type=str,
        default="snowball",
        choices=["snowball", "none"],
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="datasets",
        help="Directory to save datasets.",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 parameter.",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 parameter.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.5,
        help="BM25 parameter.",
    )
    parser.add_argument(
        "--skip_scoring",
        action="store_true",
        help="Skip scoring step.",
    )

    parser.add_argument(
        "--skip_numpy_retrieval",
        action="store_true",
        help="Skip numpy retrieval step.",
    )


    kwargs = vars(parser.parse_args())
    profile = kwargs.pop("profile")
    num_runs = kwargs.pop("num_runs")

    if profile:
        import cProfile
        import pstats

        if num_runs > 1:
            raise ValueError("Cannot profile with multiple runs.")

        cProfile.run("main(**kwargs)", filename="bm25s.prof")
        p = pstats.Stats("bm25s.prof")
        p.sort_stats("time").print_stats(50)
    else:
        for _ in range(num_runs):
            main(**kwargs)
