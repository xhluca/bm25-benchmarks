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
import bm25_pt
from transformers import AutoTokenizer

from utils.benchmark import get_max_memory_usage, Timer
import utils.huggingface
from utils.beir import (
    BASE_URL,
    clean_results_keys,
    merge_cqa_dupstack,
    postprocess_results_for_eval,
)

def get_batches(lst, batch_size=32):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

def compute_top_k_from_scores(scores, corpus=None, k=10, sorting=False, with_scores=False):
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)

    # top_n = np.argsort(scores)[::-1][:n]
    top_n = np.argpartition(scores, -k)
    # use np.take to select the last k elements
    top_n = np.take(top_n, np.argsort(scores[top_n])[-k:])
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

def main(dataset, n_threads=1, top_k=1000, batch_size=32, save_dir="datasets", result_dir="results", verbose=False):
    #### Download dataset and unzip the dataset
    data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), save_dir)

    if dataset == "cqadupstack":
        merge_cqa_dupstack(data_path)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    corpus_ids, corpus_lst = [], []
    for key, val in corpus.items():
        corpus_ids.append(key)
        corpus_lst.append(val["title"] + " " + val["text"])

    qids, queries_lst = [], []
    for key, val in queries.items():
        qids.append(key)
        queries_lst.append(val)

    print("=" * 50)
    print("Dataset: ", dataset)
    print(f"Corpus Size: {len(corpus_lst):,}")
    print(f"Queries Size: {len(queries_lst):,}")

    timer = Timer("[bm25-pt]")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    t = timer.start("Tokenize Corpus")
    tokenized_corpus = utils.huggingface.batch_tokenize(tokenizer, corpus_lst)
    timer.stop(t, show=True, n_total=len(corpus_lst))

    t = timer.start("Tokenize Queries")
    queries_tokenized = utils.huggingface.batch_tokenize(tokenizer, queries_lst)
    timer.stop(t, show=True, n_total=len(queries_lst))

    num_tokens = sum(len(doc) for doc in tokenized_corpus)
    print(f"Number of Tokens: {num_tokens:,}")
    print(f"Number of Tokens / Doc: {num_tokens / len(corpus_lst):.2f}")
    print("-" * 50)

    t = timer.start("Index")
    model = bm25_pt.BM25(tokenizer=tokenizer, device='cpu')
    model.index(corpus_lst)
    timer.stop(t)
    # we deduct the time taken to tokenize the corpus from the indexing time
    timer.results['Index']['elapsed'] -= timer.elapsed("Tokenize Corpus")
    # We can now show the time taken to index the corpus
    timer.show("Index", n_total=len(corpus_lst))

    results = []
    scores = []

    t_score = timer.start("Score")
    timer.pause("Score")
    t_query = timer.start("Query")
    batches = get_batches(queries_lst, batch_size=batch_size)
    num_batches = len(queries_lst) // batch_size + 1

    for batch in tqdm(batches, total=num_batches, desc="bm25-pt Scoring", leave=False, disable=not verbose):
        timer.resume(t_score)
        raw_scores_batch = model.score_batch(batch)
        timer.pause(t_score)
        raw_scores_batch = raw_scores_batch.cpu().numpy()
        for raw_scores in raw_scores_batch:
            result, score = compute_top_k_from_scores(
                raw_scores, corpus=corpus_ids, k=top_k, with_scores=True
            )
            results.append(result)
            scores.append(score)
    
    queried_results = np.array(results)
    queried_scores = np.array(scores)

    timer.stop(t_score)
    timer.stop(t_query)

    # we deduct the time taken to tokenize the queries from the scoring time
    timer.results['Score']['elapsed'] -= timer.elapsed("Tokenize Queries")
    timer.results['Query']['elapsed'] -= timer.elapsed("Tokenize Queries")

    # We can now show the time taken to score the queries
    timer.show("Score", n_total=len(queries_lst))
    timer.show("Query", n_total=len(queries_lst))

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
        "model": "bm25-pt",
        "dataset": dataset,
        "stemmer": "snowball",
        "tokenizer": "skl",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_threads": n_threads,
        "top_k": top_k,
        "max_mem_gb": max_mem_gb,
        "stats": {
            "num_docs": len(corpus_lst),
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
        description="Benchmark bm25-pt on a dataset.",
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
        "--num_runs", 
        type=int,
        default=1,
        help="Number of runs to repeat main."
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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for scoring.",
    )

    kwargs = vars(parser.parse_args())
    profile = kwargs.pop("profile")
    num_runs = kwargs.pop("num_runs")

    if profile:
        import cProfile
        import pstats

        if num_runs > 1:
            raise ValueError("Cannot profile with multiple runs.")

        cProfile.run("main(**kwargs)", filename="bm25pt.prof")
        p = pstats.Stats("bm25pt.prof")
        p.sort_stats("time").print_stats(50)
    else:
        for _ in range(num_runs):
            main(**kwargs)
