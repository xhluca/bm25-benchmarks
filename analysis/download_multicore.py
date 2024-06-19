# Download results from kaggle
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
import time

from kaggle.api.kaggle_api_extended import KaggleApi

ERROR_OOM = "Your notebook tried to allocate more memory than is available."
ERROR_TIMEOUT = (
    "Your notebook was stopped because it exceeded the max allowed execution duration."
)


output_base_dir = Path("./multicore_results")

# use temp folder to store results
methods_to_nb_name = {
    "bm25s": [
        "run-bm25s-multicore",
        "run-bm25s-multicore-msmarco",
    ],
    "elasticsearch": [
        "run-elasticsearch-multicore",
    ],
}


username = os.environ["KAGGLE_USERNAME"]
api = KaggleApi()
api.authenticate()

for method, notebook_names in methods_to_nb_name.items():
    for name in notebook_names:
        output_dir = output_base_dir / method
        output_dir.mkdir(parents=True, exist_ok=True)

        notebook_name = f"{username}/{name}"

        try:
            status = api.kernels_status(notebook_name)
            # if it's cancelAcknowledged, then it was cancelled after it was completed
        except Exception as e:
            print(f"Failed to get status for '{notebook_name}':\n{e}")
            continue
        else:
            print(f"Status for '{notebook_name}': {status['status']}")

        if status["hasFailureMessage"]:
            failure_message = status["failureMessage"]
            if failure_message == ERROR_OOM:
                print(f"Notebook '{notebook_name}' failed due to OOM")
            elif failure_message == ERROR_TIMEOUT:
                print(f"Notebook '{notebook_name}' failed due to timeout")
            else:
                print(
                    f"Notebook '{notebook_name}' failed with message: {failure_message}"
                )

            # save the error message to the same output dir as the results, but prefix with 'error'
            t = time.strftime("%Y%m%d-%H%M%S")
            error_file = output_dir / f"error-{name}-{t}.txt"
            with open(error_file, "w") as f:
                f.write(failure_message)
            continue

        temp_dir = tempfile.mkdtemp()

        out_files = api.kernels_output(notebook_name, temp_dir)

        print(f"Downloaded results for '{notebook_name}'")
        # only save out *.json files to the output folder
        for file in out_files:
            file = Path(file)
            if file.suffix == ".json":
                # only copy the json file, do not copy path
                shutil.copy(file, output_dir)

        # Remove temp folder
        shutil.rmtree(temp_dir)
