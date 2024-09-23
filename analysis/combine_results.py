import json
from pathlib import Path

import pandas as pd

ERROR_OOM = "Your notebook tried to allocate more memory than is available."
ERROR_TIMEOUT = (
    "Your notebook was stopped because it exceeded the max allowed execution duration."
)

results_base_dir = Path("./results")

# Go through all the methods and datasets and combine the results

# methods_to_datasets = {
#     "bm25-pt": [
#         "sub-1m",
#         "cqadupstack",
#         "webis-touche2020",
#         "nq",  # 2M
#         "msmarco",  # 8.8M
#         "hotpotqa",  # 5.2M
#         "dbpedia-entity",  # 4.6M
#         "fever",  # 5.4M
#         "climate-fever",  # 5.4M
#     ],
#     "pyserini": [
#         "sub-1m",
#         "cqadupstack",
#         "nq",  # 2M
#         "msmarco",  # 8.8M
#         "hotpotqa",  # 5.2M
#         "dbpedia-entity",  # 4.6M
#         "fever",  # 5.4M
#         "climate-fever",  # 5.4M
#     ],
#     "rank-bm25": [
#         "sub-1m",
#         "cqadupstack",
#         "nq",  # 2M
#         "msmarco",  # 8.8M
#         "hotpotqa",  # 5.2M
#         "dbpedia-entity",  # 4.6M
#         "fever",  # 5.4M
#         "climate-fever",  # 5.4M
#     ],
#     "bm25s": [
#         "sub-1m",
#         "nq",  # 2M
#         "msmarco",  # 8.8M
#         "hotpotqa",  # 5.2M
#         "dbpedia-entity",  # 4.6M
#         "fever",  # 5.4M
#         "climate-fever",  # 5.4M
#     ],
# }

model_abbreviations = {
    "bm25s": "BM25S",
    "bm25-pt": "PT",
    "pyserini": "PSRN",
    "rank-bm25": "Rank",
    "elastic-bm25": "ES",
    "pisa": "PISA",
    "retriv": "RV",
    "bm25s_jit": "BM25S+J",
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
    if r["n_threads"] > 1 or r["n_threads"] == -1:
        continue

    index_time_total = r["timing"]["index"]["elapsed"]

    # default:
    query_time_total = r["timing"]["query"]["elapsed"]

    if r["timing"].get("query_numba") is not None:
        query_time_total = r["timing"]["query_numba"]["elapsed"]

    elif r["timing"].get("query_numpy") is not None:
        query_time_total = min(query_time_total, r["timing"]["query_numpy"]["elapsed"])

    if "tokenize_corpus_(class)" in r["timing"]:
        index_time_total += r["timing"]["tokenize_corpus_(class)"]["elapsed"]

    elif "tokenize_corpus" in r["timing"]:
        index_time_total += r["timing"]["tokenize_corpus"]["elapsed"]

    if "tokenize_queries_(class)" in r["timing"]:
        query_time_total += r["timing"]["tokenize_queries_(class)"]["elapsed"]
    elif "tokenize_queries" in r["timing"]:
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
            "max_mem_gb": r.get("max_mem_gb", -1),
        }
    )

# Create another table of stats for the datasets
results_stats = {}

for r in results:
    if r["model"] != "bm25s":
        continue

    dataset = r["dataset"]
    results_stats[dataset] = {
        "num_docs": r["stats"]["num_docs"],
        "num_queries": r["stats"]["num_queries"],
        "num_tokens": r["stats"]["num_tokens"],
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

ndcg_df = df.pivot(index="dataset", columns="model", values="ndcg@10").round(4)
r_df = df.pivot(index="dataset", columns="model", values="r@1000").round(4)

# convert to wide, where columns are models, values are qps
qps_df = df.pivot(index="dataset", columns="model", values="qps").round(2)
qps_df_norm = qps_df.div(qps_df["Rank"], axis=0).round(2)
qps_df_es = qps_df.div(qps_df["ES"], axis=0).round(2)
qps_df_std = df.pivot(index="dataset", columns="model", values="qps_std").round(2)

# make a table for dps
dps_df = df.pivot(index="dataset", columns="model", values="dps").round(2)

# stats_df = pd.DataFrame(results_stats).T.map(lambda x: f"{x:,}")

# save everything as csv, markdown and latex
save_dir = Path("analysis/out")
for subdir in ["csv", "markdown", "latex"]:
    (save_dir / subdir).mkdir(parents=True, exist_ok=True)


renamed_cols = {
    "arguana": "ArguAna",
    "climate-fever": "Climate-Fever",
    "cqadupstack ": "CQADupStack",
    "dbpedia-entity": "DBpedia",
    "fever": "FEVER",
    "fiqa": "FiQA",
    "hotpotqa": "HotpotQA",
    "msmarco": "MSMARCO",
    "nfcorpus": "NFCorpus",
    "nq": "NaturalQuestions",
    "quora": "Quora",
    "scidocs": "SciDocs",
    "scifact": "SciFact",
    "trec-covid": "TREC-COVID",
    "webis-touche2020": "Touche-2020",
}

df.to_csv(save_dir / "csv" / "results.csv", index=False)
df.to_markdown(save_dir / "markdown" / "results.md", index=False)
df.to_latex(save_dir / "latex" / "results.tex", index=False, float_format="%.2f")


qps_df.index = qps_df.index.map(renamed_cols)
qps_df.to_csv(save_dir / "csv" / "qps.csv")
qps_df.to_markdown(save_dir / "markdown" / "qps.md")
qps_df.to_latex(save_dir / "latex" / "qps.tex", float_format="%.2f")

qps_df_norm.index = qps_df_norm.index.map(renamed_cols)
qps_df_norm.to_csv(save_dir / "csv" / "qps_norm.csv")
qps_df_norm.to_markdown(save_dir / "markdown" / "qps_norm.md")
qps_df_norm.to_latex(save_dir / "latex" / "qps_norm.tex", float_format="%.2f")

qps_df_es.index = qps_df_es.index.map(renamed_cols)
qps_df_es.to_csv(save_dir / "csv" / "qps_norm_es.csv")
qps_df_es.to_markdown(save_dir / "markdown" / "qps_norm_es.md")
qps_df_es.to_latex(save_dir / "latex" / "qps_norm_es.tex", float_format="%.2f")

qps_df_std.index = qps_df_std.index.map(renamed_cols)
qps_df_std.to_csv(save_dir / "csv" / "qps_std.csv")
qps_df_std.to_markdown(save_dir / "markdown" / "qps_std.md")
qps_df_std.to_latex(save_dir / "latex" / "qps_std.tex", float_format="%.4f")

dps_df.index = dps_df.index.map(renamed_cols)
dps_df.to_csv(save_dir / "csv" / "dps.csv")
dps_df.to_markdown(save_dir / "markdown" / "dps.md")
dps_df.to_latex(save_dir / "latex" / "dps.tex", float_format="%.2f")

# stats_df.to_csv(save_dir / "csv" / "stats.csv")
# stats_df.to_markdown(save_dir / "markdown" / "stats.md")
# stats_df.to_latex(save_dir  / 'latex' / "stats.tex", float_format="%.4f")

ndcg_df.index = ndcg_df.index.map(renamed_cols)
ndcg_df.to_csv(save_dir / "csv" / "ndcg.csv")
ndcg_df.to_markdown(save_dir / "markdown" / "ndcg.md")
ndcg_df.to_latex(save_dir / "latex" / "ndcg.tex", float_format="%.4f")

r_df.index = r_df.index.map(renamed_cols)
r_df.to_csv(save_dir / "csv" / "r.csv")
r_df.to_markdown(save_dir / "markdown" / "r.md")
r_df.to_latex(save_dir / "latex" / "r.tex", float_format="%.4f")

print("Results saved to analysis/out")
