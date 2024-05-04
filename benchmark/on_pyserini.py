import json
import os
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
from pyserini.search import LuceneSearcher
from pyserini.analysis import Analyzer, get_lucene_analyzer

from utils.beir import merge_cqa_dupstack

def format_beir_result_keys(beir_results):
    return {
        k.split("@")[-1]: v for k, v in beir_results.items()
    }

def convert_dict_to_text(
    d,
    sep=" ",
    key_order=("title", "subtitle", "text"),
    space_around_sep=True,
):
    """
    Convert a dictionary to a string by concatenating the values in the given order,
    separated by the given separator.

    Parameters
    ----------
    d: dict
        The dictionary to convert.

    key_order: list of strings
        The order in which to concatenate the values. If a key is not present in
        the dictionary, it is ignored.

    sep: string
        The separator to use when concatenating the values. For example, if sep is
        " ", then the values will be concatenated with a space between them.

    space_around_sep: bool
        Whether to add spaces around the separator. For example, if sep is "[SEP]", and
        space_around_sep is True, then the values will be concatenated with a space
        before and after the separator. If space_around_sep is False, then the values
        will be concatenated with no spaces before or after the separator. If sep is
        " ", then this parameter has no effect.

    Returns
    -------
    string
        The converted text. Each text is the concatenation of the values in the
        given order, separated by the given separator.
    """
    if space_around_sep and sep != " ":
        sep = f" {sep} "

    return sep.join(d[k] for k in key_order if k in d)

def convert_to_pyserini_records(
    records: List[Dict[str, str]],
    dict_to_text_fn=partial(
        convert_dict_to_text, key_order=("title", "sub_title", "text")
    ),
    n_threads=-1,
    chunk_size=1000,
):
    """
    Convert records (list of dictionaries, each one representing a document) to a list of Pyserini
    records (list of dictionaries, each one representing a document in Pyserini's JSON format). This
    function uses multiprocessing to speed up the process.

    Parameters
    ----------
    records: list of dicts
        This is a list of dictionaries, each one representing a document. This follows the format defined
        in this library. You can use the convert_records_to_texts function to convert a list of records
        to a list of texts.

    dict_to_text_fn: function
        This is a function that converts a dictionary to a string. This is used to convert each record
        to a string. The default function converts the record to a string by concatenating the values
        in the given order, separated by the given separator (see the convert_dict_to_text function for).
        You can use this parameter to customize the conversion process.

    n_threads: int, default=-1
        Number of processes to use for converting the records. If -1, the number of processes will be
        set to the number of CPU cores.

    chunk_size: int, default=1000
        The number of records to process in each chunk. This is used to control the memory usage of the
        process pool.

    Returns
    -------
    list of dicts
        This is a list of dictionaries, each one representing a document in Pyserini's JSON format.
        For example, a dictionary in this list may look like:
        - Original: {"id": "doc1", "title": "My Title", "text": "This is the text.", "sub_title": "Some subtitle"}
        - Pyserini: {"id": "doc1", "contents": "My Title Some subtitle This is the text."}

    Notes
    -----
    The docid field in the Pyserini record is set to the index of the record in the list of records.
    Note that index is different from the id field in the original record. For example, if the original
    record has id "doc1", and is the first record in the `records` list, the Pyserini record will have
    id "0", not "doc1". The original document id is not preserved in the Pyserini record.
    """
    n_threads = mp.cpu_count() if n_threads == -1 else n_threads

    def _process_single_record(rec):
        contents = dict_to_text_fn(rec)
        return {"id": rec["index"], "contents": contents}

    with mp.pool.ThreadPool(n_threads) as pool:
        new_records = pool.map(_process_single_record, records, chunksize=chunk_size)

    return new_records


def create_pyserini_json(
    records,
    directory="",
    filename="documents.json",
    overwrite=False,
    verbose=1,
    n_threads=-1,
):
    """
    Create a Pyserini JSON file from a list of records, and save it to the given directory.
    This is a thin wrapper around the convert_to_pyserini_records function.

    Parameters
    ----------
    records: list of dicts
        This is the same as the records parameter in the convert_to_pyserini_records function.

    directory: str, default=""
        Path to the directory where the JSON file will be saved.

    filename: str, default="documents.json"
        Name of the JSON file.

    overwrite: bool, default=False
        If True, the JSON file will be overwritten if it already exists. If False, the JSON file will
        not be overwritten if it already exists.

    verbose: int, default=1
        If 1, print progress messages. If 0, don't print progress messages.

    n_threads: int, default=-1
        Number of processes to use for converting the records. If -1, the number of processes will be
        set to the number of CPU cores.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename

    if path.is_file() and not overwrite:
        if verbose > 0:
            print(
                f"Pyserini JSON file already exists at '{path}', and overwrite=False. Skipping."
            )
    else:
        if verbose > 0:
            print(f"Converting dataset to Pyserini's JSON format...")
        # Convert the dataset to Pyserini's JSON (not JSONL) format
        new_records = convert_to_pyserini_records(records, n_threads=n_threads)

        with open(path, "w") as f:
            json.dump(new_records, f)

        if verbose > 0:
            print(f"Pyserini JSON file saved to {path}.")


def build_pyserini_index(
    input_dir, index_dir=None, n_threads=-1, stemmer="porter", keep_stopwords=False, python_str=None, verbose=1
):
    """
    Build a Pyserini index from a directory of JSON files. The JSON files should have the following
    format:

    {
        "id": "doc1",
        "contents": "This is the text of the document."
    }

    Parameters
    ----------
    input_dir: str
        Path to the directory containing the JSON files. All files in this directory will be indexed.
    index_dir: str
        Path to the directory where the index will be stored. If None, the index will be stored in
        the same directory as the JSON files, in a subdirectory called "index".
    stemmer: str
        Stemmer to use for indexing. Defaults to "porter". Other options are "krovetz" and "none".
    n_threads: int
        Number of threads to use for indexing. Defaults to 1. If -1, the number of threads will be
        set to the number of CPU cores.
    python_str: str
        String of Python executable to use for indexing. Defaults to "python". If you are using a
        virtual environment, you may need to specify the full path to the Python executable, e.g.
        "/path/to/venv/bin/python".
    verbose: int
        Verbosity level. Defaults to 1. If 0, no output will be printed. If 1, the command that is
        run will be printed.
    keep_stopwords: bool
        Whether to keep stopwords in the index. Defaults to False.
    Returns
    -------
    subprocess.CompletedProcess
        The result of the subprocess run. See the Python documentation for more details.
    """
    if n_threads < 1:
        import multiprocessing

        n_threads = multiprocessing.cpu_count()

    if python_str is None:
        python_str = sys.executable

    input_dir = Path(input_dir)

    if index_dir is None:
        index_dir = input_dir / "index"

    if stemmer in [None, False]:
        stemmer = "none"
    if stemmer not in ["porter", "krovetz", "none"]:
        raise ValueError(f"Invalid stemmer: {stemmer}")

    command = f"""
    {python_str} -m pyserini.index.lucene \\
    --collection JsonCollection \\
    --input {input_dir} \\
    --index {index_dir} \\
    --generator DefaultLuceneDocumentGenerator \\
    --stemmer {stemmer} \\
    --threads {n_threads} \\
    --storePositions \\
    --storeDocvectors \\
    --storeRaw"""

    if keep_stopwords:
        command += "\\\n    --keepStopwords"
    
    if verbose > 0:
        print("Running command:")
        print(command)
        print("Output:")
        out = subprocess.run(command, shell=True)
        print("Done.")
    else:
        out = subprocess.run(
            command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )

    return out


def main(dataset, save_dir="datasets", result_dir="results", n_threads=1, top_k=1000):
    warnings.filterwarnings("ignore", category=UserWarning)

    #### Download dataset and unzip the dataset
    base_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

    url = base_url.format(dataset)
    data_dir = Path(save_dir)
    data_path = beir.util.download_and_unzip(url, str(data_dir))
    if dataset == "cqadupstack":
            merge_cqa_dupstack(data_path)
        
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    num_docs = len(corpus)
    corpus_records = [
        {'id': key, 'contents': val['title'] + " " + val['text']} for key, val in corpus.items()
    ]
    del corpus

    queries_lst = []
    qids = []
    for key, val in queries.items():
        queries_lst.append(val)
        qids.append(key)


    pyserini_data_dir = data_dir / dataset / "pyserini"
    pyserini_data_dir.mkdir(parents=True, exist_ok=True)
    #### Convert the dataset to Pyserini's JSON format
    with open(pyserini_data_dir / "corpus.json", "w") as f:
        json.dump(corpus_records, f)

    del corpus_records
    t0 = time.time()
    build_pyserini_index(input_dir=pyserini_data_dir, n_threads=n_threads)
    time_index = time.time() - t0

    print('='*50)
    print(f"[Pyserini] Index: {time_index:.4f}s ({num_docs / time_index:.2f}/s)")

    # now, run query and get score as well
    
    # lucene_analyzer = get_lucene_analyzer(stemmer='snowball', stopwords=True)
    # analyzer = Analyzer(lucene_analyzer)

    searcher = LuceneSearcher(str(pyserini_data_dir / "index"))
    # searcher.set_analyzer(analyzer=lucene_analyzer)
    searcher.set_bm25(k1=1.5, b=0.75)
    # ################### BEIR BENCHMARKING HERE ###################
    # results's format: {query_id: {doc_id: score, doc_id: score, ...}, ...}
    k_values = [1,10,100,1000]

    t0 = time.time()
    hits = searcher.batch_search(queries_lst, qids=qids, k=top_k, threads=n_threads)
    time_search = time.time() - t0
    print(f"[Pyserini] Query: {time_search:.4f}s ({len(queries_lst) / time_search:.2f}/s)")
    print('-'*50)

    results = {
        qid: {hit.docid: hit.score for hit in hit_list} for qid, hit_list in hits.items()
    }

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)
    print(ndcg)
    print(recall)
    print(precision)

    # Save everything to json
    save_dict = {
        "model": "pyserini",
        "dataset": dataset,
        "stemmer": "porter",
        "tokenizer": "lucene",
        "method": "lucene",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_threads": n_threads,
        "top_k": top_k,
        "stats": {
            "num_docs": num_docs,
            "num_queries": len(queries_lst),
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

    result_dir = Path(result_dir) / "pyserini"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(result_dir) / f"{dataset}-{os.urandom(8).hex()}.json"
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark pyserini on a dataset.",
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

    kwargs = vars(parser.parse_args())
    main(**kwargs)