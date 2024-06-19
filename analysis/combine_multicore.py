import json
from pathlib import Path

import pandas as pd

ERROR_OOM = "Your notebook tried to allocate more memory than is available."
ERROR_TIMEOUT = (
    "Your notebook was stopped because it exceeded the max allowed execution duration."
)

results_base_dir = Path("./multicore_results")
save_dir = Path("analysis/out/multicore")

# Go through all the methods and datasets and combine the results

model_abbreviations = {
    "bm25s": "BM25S",
    "bm25-pt": "PT",
    "pyserini": "PSRN",
    "rank-bm25": "Rank",
    "elastic-bm25": "ES",
}

removed_models = [
    # 'pyserini',
    # "bm25s"
]

# Load all results
results = []
# get all file (in dir or subdir) with the pattern *-*.json
for file in results_base_dir.rglob("*-*.json"):
    with open(file, "r") as f:
        results.append(json.load(f))

results_processed = []

# Process them
for r in results:
    index_time_total = r["timing"]["index"]["elapsed"]
    query_time_total = r["timing"]["query"]["elapsed"]
    if r['timing'].get('query_np') is not None:
        query_time_total = min(query_time_total, r['timing']['query_np']['elapsed'])

    if "tokenize_corpus" in r["timing"]:
        index_time_total += r["timing"]["tokenize_corpus"]["elapsed"]

    if "tokenize_queries" in r["timing"]:
        query_time_total += r["timing"]["tokenize_queries"]["elapsed"]

    n_docs = r["stats"]["num_docs"]
    n_queries = r["stats"]["num_queries"]

    if "ndcg" not in r:
        r["ndcg"] = r["scores"]["ndcg"]
    if "recall" not in r:
        r["recall"] = r["scores"]["recall"]

    # removed_models

    if r["model"] in removed_models:
        continue

    results_processed.append(
        {
            "model": model_abbreviations[r["model"]],
            "dataset": r["dataset"],
            "ndcg@10": r["ndcg"]["10"],
            "r@1000": r["recall"]["1000"],
            "qps": n_queries / query_time_total,
            "dps": n_docs / index_time_total,
            'max_mem_gb': r.get('max_mem_gb', -1)
        }
    )

# Create another table of stats for the datasets
results_stats = {}

for r in results:
    if r['model'] != 'bm25s':
        continue
    
    dataset = r['dataset']
    results_stats[dataset] = {
        'num_docs': r['stats']['num_docs'],
        'num_queries': r['stats']['num_queries'],
        'num_tokens': r['stats']['num_tokens'],
    }

# Now, let's combine all the results into a single DataFrame
df = pd.DataFrame(results_processed)

# for duplicate values wrt model/dataset, we take the mean and std
df = (
    df.groupby(["model", "dataset"])
    .agg(
        {
            "ndcg@10": ["mean"],
            "r@1000": ["mean"],
            "qps": ["mean", "std"],
            "dps": ["mean", "std"],
        }
    )
    .reset_index()
)

# merge columns for easier access, remove _mean from column names
df.columns = [
    "_".join(col).strip() if col[1] != "" else col[0] for col in df.columns.values
]
df.columns = [col.replace("_mean", "") for col in df.columns.values]

# remove pyserini column for now
# convert to wide, where columns are models, values are qps
dps_df = df.pivot(index="dataset", columns="model", values="dps").round(2)
qps_df = df.pivot(index="dataset", columns="model", values="qps").round(2)
qps_df_es = qps_df.div(qps_df["ES"], axis=0).round(2)
qps_df_std = df.pivot(index="dataset", columns="model", values="qps_std").round(2)

# stats_df = pd.DataFrame(results_stats).T.map(lambda x: f"{x:,}")

# save everything as csv, markdown and latex
for subdir in ["csv", "markdown", "latex"]:
    (save_dir / subdir).mkdir(parents=True, exist_ok=True)

df.to_csv(save_dir / "csv" / "results.csv", index=False)
df.to_markdown(save_dir / "markdown" / "results.md", index=False)
df.to_latex(save_dir / 'latex' / "results.tex", index=False, float_format="%.2f")

qps_df.to_csv(save_dir / "csv" / "qps.csv")
qps_df.to_markdown(save_dir / "markdown" / "qps.md")
qps_df.to_latex(save_dir  / 'latex' / "qps.tex", float_format="%.2f")

qps_df_es.to_csv(save_dir / "csv" / "qps_norm_es.csv")
qps_df_es.to_markdown(save_dir / "markdown" / "qps_norm_es.md")
qps_df_es.to_latex(save_dir  / 'latex' / "qps_norm_es.tex", float_format="%.2f")

qps_df_std.to_csv(save_dir / "csv" / "qps_std.csv")
qps_df_std.to_markdown(save_dir / "markdown" / "qps_std.md")
qps_df_std.to_latex(save_dir  / 'latex' / "qps_std.tex", float_format="%.4f")

dps_df.to_csv(save_dir / "csv" / "dps.csv")
dps_df.to_markdown(save_dir / "markdown" / "dps.md")
dps_df.to_latex(save_dir / 'latex' / "dps.tex", float_format="%.2f")

print(f"Results saved to {save_dir}")
