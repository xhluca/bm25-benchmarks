import json
import math
import os
from pathlib import Path
import time

import numpy as np
from tqdm.auto import tqdm
import Stemmer
from numba import njit

import bm25s
from bm25s.utils.benchmark import get_max_memory_usage, Timer
from bm25s.utils.beir import (
    clean_results_keys,
    download_dataset,
    load_corpus,
    load_queries,
    postprocess_results_for_eval,
)


def load_qrels_dict(dataset, split="test", save_dir="datasets"):
    qrels_path = Path(save_dir) / dataset / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found at {qrels_path}")

    qrels = {}
    with open(qrels_path, "r") as f:
        next(f)
        for line in f:
            qid, cid, score = line.strip().split("\t")
            qrels.setdefault(qid, {})[cid] = int(score)
    return qrels


def evaluate(qrels, results, k_values, ignore_identical_ids=True):
    ndcg, _map, recall, precision = {}, {}, {}, {}
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    num_queries = len(qrels)
    if num_queries == 0:
        raise ValueError("Cannot evaluate with no qrels.")

    for qid, rels in qrels.items():
        relevant = {docid: rel for docid, rel in rels.items() if rel > 0}
        ranked = sorted(results.get(qid, {}).items(), key=lambda item: item[1], reverse=True)
        if ignore_identical_ids:
            ranked = [(docid, score) for docid, score in ranked if docid != qid]

        ideal_rels = sorted(relevant.values(), reverse=True)
        total_relevant = len(relevant)

        for k in k_values:
            top_k = ranked[:k]
            hits = 0
            ap_sum = 0.0
            dcg = 0.0

            for rank, (docid, _) in enumerate(top_k, start=1):
                rel = relevant.get(docid, 0)
                if rel > 0:
                    hits += 1
                    ap_sum += hits / rank
                    dcg += (2**rel - 1) / math.log2(rank + 1)

            idcg = sum(
                (2**rel - 1) / math.log2(rank + 1)
                for rank, rel in enumerate(ideal_rels[:k], start=1)
            )
            ndcg[f"NDCG@{k}"] += dcg / idcg if idcg > 0 else 0.0
            _map[f"MAP@{k}"] += ap_sum / min(total_relevant, k) if total_relevant > 0 else 0.0
            recall[f"Recall@{k}"] += hits / total_relevant if total_relevant > 0 else 0.0
            precision[f"P@{k}"] += hits / k

    for metric in (ndcg, _map, recall, precision):
        for key in metric:
            metric[key] = round(metric[key] / num_queries, 5)

    return ndcg, _map, recall, precision


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
    scorers=None,
    backends=None,
    use_function_tokenization=False,
):
    if scorers is None:
        scorers = ["legacy", "jit"]
    if backends is None:
        backends = ["numba"]
    #### Download dataset and unzip the dataset
    data_path = Path(save_dir) / dataset
    corpus_file = data_path / "corpus.jsonl"

    if not corpus_file.exists():
        data_path = download_dataset(dataset, save_dir=save_dir)

    if dataset == "msmarco":
        split = "dev"
    else:
        split = "test"

    corpus = load_corpus(dataset, save_dir=save_dir)
    queries = load_queries(dataset, save_dir=save_dir)
    qrels = load_qrels_dict(dataset, split=split, save_dir=save_dir)
    num_docs = len(corpus)

    corpus_ids, corpus_lst = [], []
    for key, val in corpus.items():
        corpus_ids.append(key)
        corpus_lst.append(f"{val.get('title') or ''} {val['text']}".strip())

    corpus_ids = np.array(corpus_ids)
    del corpus

    qids, queries_lst = [], []
    for key, val in queries.items():
        qids.append(key)
        if isinstance(val, dict):
            queries_lst.append(val["text"])
        else:
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

    # tokenizer class
    tokenizer = bm25s.tokenization.Tokenizer(
        stopwords=stopwords, 
        stemmer=stemmer,
    )

    t = timer.start("Tokenize Corpus (class)")
    corpus_tokenized = tokenizer.tokenize(corpus_lst, update_vocab=True, return_as="tuple")
    timer.stop(t, show=True, n_total=num_docs)

    t = timer.start("Tokenize Queries (class)")
    queries_ids = tokenizer.tokenize(queries_lst, update_vocab=False, return_as="ids")
    queries_tokenized = tokenizer.tokenize(queries_lst, update_vocab=False, return_as="string")
    timer.stop(t, show=True, n_total=len(queries_lst))

    if use_function_tokenization:
        t = timer.start("Tokenize Corpus (function)")
        corpus_tokenized_fn = bm25s.tokenize(
            corpus_lst,
            stopwords=stopwords,
            stemmer=stemmer,
            leave=False,
            return_ids=True,
        )
        timer.stop(t, show=True, n_total=num_docs)

        t = timer.start("Tokenize Queries (function)")
        queries_tokenized_fn = bm25s.tokenize(
            queries_lst,
            stopwords=stopwords,
            stemmer=stemmer,
            leave=False,
            return_ids=False,
        )
        timer.stop(t, show=True, n_total=len(queries_lst))

    del corpus_lst

    num_tokens = sum(len(doc) for doc in corpus_tokenized[0])
    num_query_tokens = sum(len(q) for q in queries_tokenized)
    num_queries = len(queries_lst)
    print(f"Number of Corpus Tokens: {num_tokens:,}")
    print(f"Number of Tokens / Doc: {num_tokens / num_docs:.2f}")
    print(f"Number of Tokens / Query: {num_query_tokens / num_queries:.2f}")
    print("-" * 50)

    # auto_compile is only available in dev version
    try:
        model = bm25s.BM25(method=method, k1=k1, b=b, delta=delta, auto_compile=False)
    except TypeError:
        model = bm25s.BM25(method=method, k1=k1, b=b, delta=delta)

    # activate the csc numba backend and warmup (if available)
    if hasattr(model, 'activate_numba_csc'):
        model.activate_numba_csc()
        model.warmup_numba_csc()
        print("Using Numba CSC Backend for Indexing")
    
    t = timer.start("Index")
    model.index(corpus_tokenized, leave_progress=False)
    timer.stop(t, show=True, n_total=num_docs)
    _compute_relevance_from_scores = model._compute_relevance_from_scores

    if "uncompiled" in scorers:
        t = timer.start("Score (uncompiled)")
        for q in tqdm(queries_tokenized, desc="BM25S Scoring", leave=False):
            model.get_scores(q)
        timer.stop(t, show=True, n_total=len(queries_lst))

    if "legacy" in scorers:
        model._compute_relevance_from_scores = bm25s.scoring._compute_relevance_from_scores_legacy
        t = timer.start("Score (legacy)")
        for q in tqdm(queries_tokenized, desc="BM25S Scoring (legacy)", leave=False):
            model.get_scores(q)
        timer.stop(t, show=True, n_total=len(queries_lst))

    if "jit" in scorers:
        # Use njit and warmup (if available)
        if hasattr(model, 'activate_numba_scorer'):
            model.activate_numba_scorer()
            if hasattr(model, 'warmup_numba_scorer'):
                model.warmup_numba_scorer()

            t = timer.start("Score (jit)")
            for q in tqdm(queries_tokenized, desc="BM25S Scoring (jit)", leave=False):
                model.get_scores(q)
            timer.stop(t, show=True, n_total=len(queries_lst))
        else:
            print("Skipping jit scorer: activate_numba_scorer not available")

    # Use njit and warmup (if available)
    if hasattr(model, 'activate_numba_scorer'):
        model.activate_numba_scorer()
        if hasattr(model, 'warmup_numba_scorer'):
            model.warmup_numba_scorer()
    # # reset back to original
    # model._compute_relevance_from_scores = _compute_relevance_from_scores

    ############## BENCHMARKING BEIR HERE ##############
    # for v1, v2 in zip(queries_ids, queries_tokenized):
    #     if v1 != v2 and np.any(model.get_scores(v1) != model.get_scores(v2)):
    #         breakpoint()

    queried_results = None
    queried_scores = None

    if "jax" in backends:
        t = timer.start("Query (jax)")
        queried_results, queried_scores = model.retrieve(
            queries_tokenized,
            corpus=corpus_ids,
            k=top_k,
            return_as="tuple",
            n_threads=n_threads,
            backend_selection="jax",
        )
        timer.stop(t, show=True, n_total=len(queries_lst))

    if "numba" in backends:
        # warmup
        model.backend = "numba"
        model.retrieve(queries_ids[:2])
        t = timer.start("Query (numba)")
        queried_results_nbs, queried_scores_nbs = model.retrieve(
            query_tokens=queries_ids,
            corpus=corpus_ids,
            k=top_k,
            return_as="tuple",
            n_threads=n_threads
        )
        timer.stop(t, show=True, n_total=len(queries_lst))

        if queried_scores is not None:
            assert np.allclose(queried_scores, queried_scores_nbs, atol=1e-6)

        queried_results = queried_results_nbs
        queried_scores = queried_scores_nbs

    if "numpy" in backends:
        model.backend = "numpy"
        t = timer.start("Query (numpy)")
        queried_results_np, queried_scores_np = model.retrieve(
            queries_tokenized,
            corpus=corpus_ids,
            k=top_k,
            return_as="tuple",
            n_threads=n_threads,
            backend_selection="numpy",
            sorted=True,
        )
        timer.stop(t, show=True, n_total=len(queries_lst))

        if queried_scores is not None:
            assert queried_scores.shape == queried_scores_np.shape
            assert np.allclose(queried_scores, queried_scores_np, atol=1e-6)

        queried_results = queried_results_np
        queried_scores = queried_scores_np
    
    results_dict = postprocess_results_for_eval(queried_results, queried_scores, qids)
    ndcg, _map, recall, precision = evaluate(qrels, results_dict, [1, 10, 100, 1000])

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
        "version": bm25s.__version__,
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

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(result_dir) / f"{dataset}-{os.urandom(8).hex()}.json"
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"Results saved to {save_path}")

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
        "--scorers",
        nargs="+",
        default=["legacy", "jit"],
        choices=["uncompiled", "legacy", "jit"],
        help="Scorers to include for benchmarking.",
    )

    parser.add_argument(
        "--backends",
        nargs="+",
        default=["numba"],
        choices=["jax", "numba", "numpy"],
        help="Backends to include for retrieval.",
    )

    parser.add_argument(
        "--use_function_tokenization",
        action="store_true",
        help="Also run function-based tokenization (bm25s.tokenize) for comparison.",
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
