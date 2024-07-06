# Download results from kaggle
from dataclasses import dataclass
import fnmatch
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import requests
from kaggle.api.kaggle_api_extended import KaggleApi

from utils import kernels_output

ERROR_OOM = 'Your notebook tried to allocate more memory than is available.'
ERROR_TIMEOUT = 'Your notebook was stopped because it exceeded the max allowed execution duration.'


output_base_dir = Path("./results")


# use temp folder to store results
method_to_notebooks = {
    # 'bm25-pt': [
    #     'xhlulu/benchmark-bm25-pt-sub-1m',
    #     'xhlulu/benchmark-bm25-pt-cqadupstack',
    #     'xhlulu/benchmark-bm25-pt-webis-touche2020',
    #     'xhlulu/benchmark-bm25-pt-nq',
    #     'xhlulu/benchmark-bm25-pt-msmarco',
    #     'xhlulu/benchmark-bm25-pt-hotpotqa', 
    #     'xhlulu/benchmark-bm25-pt-dbpedia-entity', 
    #     'xhlulu/benchmark-bm25-pt-fever',  
    #     'xhlulu/benchmark-bm25-pt-climate-fever',  
    # ],
    # 'pyserini': [
    #     'xhlulu/benchmark-pyserini-sub-1m',
    #     'xhlulu/benchmark-pyserini-cqadupstack',
    #     'xhlulu/benchmark-pyserini-nq',
    #     'xhlulu/benchmark-pyserini-msmarco',
    #     'xhlulu/benchmark-pyserini-hotpotqa', 
    #     'xhlulu/benchmark-pyserini-dbpedia-entity', 
    #     'xhlulu/benchmark-pyserini-fever',  
    #     'xhlulu/benchmark-pyserini-climate-fever',  
    # ],
    # 'rank-bm25': [
    #     'xhlulu/benchmark-rank-bm25-sub-1m',
    #     'xhlulu/benchmark-rank-bm25-cqadupstack',
    #     'xhlulu/benchmark-rank-bm25-nq',
    #     'xhlulu/benchmark-rank-bm25-msmarco',
    #     'xhlulu/benchmark-rank-bm25-hotpotqa', 
    #     'xhlulu/benchmark-rank-bm25-dbpedia-entity', 
    #     'xhlulu/benchmark-rank-bm25-fever',  
    #     'xhlulu/benchmark-rank-bm25-climate-fever',  
    # ],
    # 'bm25s': [
    #     'xhlulu/benchmark-bm25s-sub-1m',
    #     'xhlulu/benchmark-bm25s-nq',
    #     'xhlulu/benchmark-bm25s-msmarco',
    #     'xhlulu/benchmark-bm25s-hotpotqa', 
    #     'xhlulu/benchmark-bm25s-dbpedia-entity', 
    #     'xhlulu/benchmark-bm25s-fever',  
    #     'xhlulu/benchmark-bm25s-climate-fever',  
    # ],
    # 'elasticsearch': [
    #     'xhlulu/benchmark-elasticsearch-sub-1m',
    #     'xhlulu/benchmark-elasticsearch-nq',
    #     'xhlulu/benchmark-elasticsearch-msmarco',
    #     'xhlulu/benchmark-elasticsearch-hotpotqa', 
    #     'xhlulu/benchmark-elasticsearch-dbpedia-entity', 
    #     'xhlulu/benchmark-elasticsearch-fever',
    # ],
    "pisa": [
        # "smac2048/pisa-nq",
        # "smac2048/pisa-rest",
        # "smac2048/pisa-dbpedia-entity",
        # "smac2048/pisa-climate-fever",
        # "smac2048/pisa-hotpotqa",
        # "smac2048/pisa-fever",
        # "smac2048/pisa-msmarco",
        "smac2048/pisa-cqadupstack",
    ]
}


username = os.environ['KAGGLE_USERNAME']
api = KaggleApi()
api.authenticate()

for method, notebooks in method_to_notebooks.items():
    for notebook_name in notebooks:   
        output_dir = output_base_dir / method
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            status = api.kernels_status(notebook_name)
            # if it's cancelAcknowledged, then it was cancelled after it was completed
        except Exception as e:
            print(f"Failed to get status for '{notebook_name}':\n{e}")
            continue
        else:
            print(f"Status for '{notebook_name}': {status['status']}")

        if status['hasFailureMessage']:
            failure_message = status['failureMessage']
            if failure_message == ERROR_OOM:
                print(f"Notebook '{notebook_name}' failed due to OOM")
            elif failure_message == ERROR_TIMEOUT:
                print(f"Notebook '{notebook_name}' failed due to timeout")
            else:
                print(f"Notebook '{notebook_name}' failed with message: {failure_message}")

            # save the error message to the same output dir as the results, but prefix with 'error'
            t = time.strftime('%Y%m%d-%H%M%S')
            error_file = output_dir / f'error-{notebook_name.replace("/", "_")}-{t}.txt'
            with open(error_file, 'w') as f:
                f.write(failure_message)
            continue
            
        temp_dir = tempfile.mkdtemp()

        out_files = kernels_output(api, notebook_name, temp_dir, allowed_patterns=['*.json'])
    
        print(f"Downloaded results for '{notebook_name}'")    
        # only save out *.json files to the output folder
        for file in out_files:
            file = Path(file)
            if file.suffix == '.json':
                # only copy the json file, do not copy path
                shutil.copy(file, output_dir)

        # Remove temp folder
        shutil.rmtree(temp_dir)
