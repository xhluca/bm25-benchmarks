import pandas as pd
import json
import os
from glob import glob
from pathlib import Path
import time
from typing import Optional
import warnings
from functools import partial
import subprocess
import json
from pathlib import Path
import sys
from typing import List, Dict
import multiprocessing as mp

from tqdm.auto import tqdm
import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from pyterrier_pisa import PisaIndex
import pyterrier as pt ; pt.init()

from utils.beir import merge_cqa_dupstack

def format_beir_result_keys(beir_results):
    return {
        k.split("@")[-1]: v for k, v in beir_results.items()
    }


def build_pisa_index(
    corpus_records, index_dir, n_threads=1, stemmer="porter2", keep_stopwords=False, python_str=None, verbose=1, k1=1.2, b=0.75
):
    """
    Build a PISA index from a list of corpus_records.
    """
    if n_threads < 1:
        import multiprocessing

        n_threads = multiprocessing.cpu_count()

    if stemmer in [None, False]:
        stemmer = "none"

    index = PisaIndex(str(index_dir), stemmer=stemmer, threads=n_threads)

    index.indexer(mode="overwrite").index(corpus_records)

    return index.bm25(k1=k1, b=b, threads=n_threads)


def main(dataset, save_dir="datasets", result_dir="results", n_threads=1, top_k=1000, k1=1.2, b=0.75):
    warnings.filterwarnings("ignore", category=UserWarning)

    #### Download dataset and unzip the dataset
    base_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

    url = base_url.format(dataset)
    data_dir = Path(save_dir)
    data_path = beir.util.download_and_unzip(url, str(data_dir))
    if dataset == "cqadupstack":
            # merge and clean up old files
            if os.path.exists(f'{data_path}.zip'):
                os.remove(f'{data_path}.zip')
            merge_cqa_dupstack(data_path)
            for hgx in glob(f'{data_path}/*/*.jsonl'):
                os.remove(hgx)
    
    if dataset == "msmarco":
        split = "dev"
    else:
        split = "test"
    
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    
    num_docs = len(corpus)
    corpus_records = [
        {'docno': key, 'text': val['title'] + " " + val['text']} for key, val in corpus.items()
    ]
    del corpus

    query_frame = pd.DataFrame(queries.items(), columns=['qid', 'query'])

    t0 = time.time()
    bm25 = build_pisa_index(corpus_records=corpus_records, index_dir=data_dir/dataset/'index.pisa', n_threads=n_threads, k1=k1, b=b)
    time_index = time.time() - t0

    print('='*50)
    print(f"[PISA] Index: {time_index:.4f}s ({num_docs / time_index:.2f}/s)")

    # now, run query and get score as well
    
    # lucene_analyzer = get_lucene_analyzer(stemmer='snowball', stopwords=True)
    # analyzer = Analyzer(lucene_analyzer)

    # ################### BEIR BENCHMARKING HERE ###################
    # results's format: {query_id: {doc_id: score, doc_id: score, ...}, ...}
    k_values = [1,10,100,1000]

    t0 = time.time()
    bm25.num_results = top_k
    bm25.threads = n_threads
    hits = bm25(query_frame)
    time_search = time.time() - t0
    print(f"[PISA] Query: {time_search:.4f}s ({len(query_frame) / time_search:.2f}/s)")
    print('-'*50)

    results = {}
    for qid, docno, score in hits[['qid', 'docno', 'score']].itertuples(index=False):
        if qid not in results:
            results[qid] = {}
        results[qid][docno] = score

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)
    print(ndcg)
    print(recall)
    print(precision)

    # Save everything to json
    save_dict = {
        "model": "pisa",
        "dataset": dataset,
        "stemmer": "porter",
        "tokenizer": "pisa",
        "method": "pisa",
        "k1": k1,
        "b": b,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_threads": n_threads,
        "top_k": top_k,
        "stats": {
            "num_docs": num_docs,
            "num_queries": len(query_frame),
        },
        "timing": {
            "index": {"elapsed": round(time_index, 4)},
            "query": {"elapsed": round(time_search, 4)},
        },
        "ndcg": format_beir_result_keys(ndcg),
        "map": format_beir_result_keys(_map),
        "recall": format_beir_result_keys(recall),
        "precision": format_beir_result_keys(precision),
    }

    result_dir = Path(result_dir) / "pisa"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(result_dir) / f"{dataset}-{os.urandom(8).hex()}.json"
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark pisa on a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="fiqa",
        help="Dataset to benchmark on.",
    )

    parser.add_argument(
        "-n", "--n_threads",
        type=int,
        default=1,
        help="Number of jobs to run in parallel.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="Number of top-k documents to retrieve.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="datasets",
        help="Directory to download datasets.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Directory to save results.",
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
    main(**kwargs)
