import json
from pathlib import Path

import pandas as pd

def save_as_csv_latex_markdown(df, save_dir, name):
    (save_dir / "csv").mkdir(parents=True, exist_ok=True)
    (save_dir / "latex").mkdir(parents=True, exist_ok=True)
    (save_dir / "markdown").mkdir(parents=True, exist_ok=True)
    
    df.to_csv(save_dir / "csv" / f"{name}.csv", index=False)
    print(f"Saved {save_dir}/csv/{name}.csv")
    df.to_latex(save_dir / "latex" / f"{name}.tex", float_format="%.1f", index=False)
    print(f"Saved {name}.tex")
    df.to_markdown(save_dir / "markdown" / f"{name}.md", index=False)
    print(f"Saved {name}.md")

ERROR_OOM = "Your notebook tried to allocate more memory than is available."
ERROR_TIMEOUT = (
    "Your notebook was stopped because it exceeded the max allowed execution duration."
)

results_base_dir = Path("./comparison_results")
save_dir = Path("./analysis/comparison_results")

model_abbreviations = {
    "bm25s": "BM25S",
    "bm25-pt": "PT",
    "pyserini": "PSRN",
    "rank-bm25": "Rank",
    "elastic-bm25": "ES",
    "pisa": "PISA",
}


old_default_params = {
    'pyserini': {'k1': 1.5, 'b': 0.75},
    'rank-bm25': {'k1': 1.5, 'b': 0.75},
    'bm25-pt': {'k1': 1.5, 'b': 0.75},
    'bm25s': {'k1': 1.5, 'b': 0.75},
    'elastic-bm25': {'k1': 1.5, 'b': 0.75},
    'pisa': {'k1': 1.2, 'b': 0.75},
}

# ['arguana', 'climate-fever', 'cqadupstack', 'dbpedia-entity', 'fever',
#        'fiqa', 'hotpotqa', 'msmarco', 'nfcorpus', 'nq', 'quora', 'scidocs',
#        'scifact', 'trec-covid', 'webis-touche2020']
dataset_shorthands = {
    'nq': 'NQ',
    'msmarco': 'MS',
    'hotpotqa': 'HP',
    'dbpedia-entity': 'DB',
    'fever': 'FV',
    'climate-fever': 'CF',
    'cqadupstack': 'CD',
    'arguana': 'AG',
    'trec-covid': 'TC',
    'webis-touche2020': 'WT',
    'scifact': 'SF',
    'scidocs': 'SD',
    'quora': 'QR',
    'nfcorpus': 'NF',
    'fiqa': 'FQ',
}

index_shorthands = {
    'stopwords': "Stop",
    'stemmer': "Stem",
}

removed_models = [
    # 'pyserini',
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
            "k1": r.get("k1", old_default_params[r["model"]]["k1"]),
            "b": r.get("b", old_default_params[r["model"]]["b"]),
            "delta": r.get("delta", 0.5),
            "method": r.get("method", "N/A"),
            "stopwords": r.get("stopwords", "N/A"),
            "stemmer": r.get("stemmer", "N/A"),
        }
    )

# Create a DataFrame
df = pd.DataFrame(results_processed)
df["stopwords"] = df["stopwords"].fillna("None")
df["stemmer"] = df["stemmer"].fillna("None")
df['dataset'] = df['dataset'].map(dataset_shorthands)
df['stemmer'] = df['stemmer'].replace({'snowball': 'Snow.'})
df['stopwords'] = df['stopwords'].replace({'en': 'Eng.'})
df["ndcg@10"] = df["ndcg@10"] * 100
df["r@1000"] = df["r@1000"] * 100
df["ndcg@10"] = df["ndcg@10"].round(1)
df["r@1000"] = df["r@1000"].round(1)
df['method'] = df['method'].map({
    'lucene': 'Lucene', 'atire': 'ATIRE', 'bm25l': 'BM25L', 'bm25+': 'BM25+', 'robertson': 'Robertson'
})
# First, we are interested in the effect of stopwords/stemmer. We will pick k1=1.5, b=0.75, model=BM25S, method=lucene
# and compare the effect of stopwords/stemmer on the datasets

df_tok = df[
    (df["model"] == "BM25S")
    & (df["k1"] == 1.5)
    & (df["b"] == 0.75)
    & (df["method"] == "Lucene")
].copy()

# fill stemmer and stopwords cols with 'None' if it's NaN

# we now want the values to be the ndcg@10, the columns are datasets, and the index is the stopwords/stemmer
df_tok_table = df_tok.pivot_table(
    index=["stopwords", "stemmer"], columns="dataset", values="ndcg@10"
)
df_tok_table.index.names = [index_shorthands[col] for col in df_tok_table.index.names]
df_tok_table["Avg."] = df_tok_table.mean(axis=1).astype(float).round(1)

# Avg should be the first column
df_tok_table = df_tok_table[
    ["Avg."] + [col for col in df_tok_table.columns if col != "Avg."]
]
df_tok_table.reset_index(inplace=True)
# replace None with No, snowball with "Snow", "en" with "Eng"

save_as_csv_latex_markdown(df_tok_table, save_dir, "tokenizer_effect")


# Now, let's look at all cases where stopwords/stemmer are en/snowball, BM25S

df_var = df[
    (df["model"] == "BM25S")
    & ((df["stopwords"] == "Eng.") 
       & (df["stemmer"] == "Snow."))
].copy()

# we also want to include other non-bm25s models
df_alt_models = df[df["model"] != "BM25S"].copy()
# replace method with model
df_alt_models['method'] = df_alt_models['model']

# combine the two dataframes
df_var = pd.concat([df_var, df_alt_models])

# We want the columns to be the dataset, values to be ndcg@10, and the index to be k1, b, method
df_var_table = df_var.pivot_table(
    index=["k1", "b", "method"], columns="dataset", values="ndcg@10"
)
# get average, but fill cells with NaN with the average over the columns
df_var_table["Avg."] = df_var_table.mean(axis=1).astype(float).round(1)
is_missing = df_var_table.isnull().sum(axis=1) > 0
df_var_table.loc[is_missing, 'Avg.'] = None

# Avg should be the first column
df_var_table = df_var_table[
    ["Avg."] + [col for col in df_var_table.columns if col != "Avg."]
]
# replace NaN with --
df_var_table = df_var_table.fillna("--")

df_var_table.reset_index(inplace=True)
# cast to str
df_var_table['k1'] = df_var_table['k1'].astype(str)
df_var_table['b'] = df_var_table['b'].astype(str)
save_as_csv_latex_markdown(df_var_table, save_dir=save_dir, name="bm25s_variants")

# Now, do the same thing for recall@1000
df_var_table_recall = df_var.pivot_table(
    index=["k1", "b", "method"], columns="dataset", values="r@1000"
)
# get average
df_var_table_recall["Avg."] = df_var_table_recall.mean(axis=1).astype(float).round(1)
is_missing = df_var_table_recall.isnull().sum(axis=1) > 0
df_var_table_recall.loc[is_missing, 'Avg.'] = None
# Avg should be the first column
df_var_table_recall = df_var_table_recall[
    ["Avg."] + [col for col in df_var_table_recall.columns if col != "Avg."]
]
# replace NaN with --
df_var_table_recall = df_var_table_recall.fillna("--")

df_var_table_recall.reset_index(inplace=True)

# cast to str
df_var_table_recall['k1'] = df_var_table_recall['k1'].astype(str)
df_var_table_recall['b'] = df_var_table_recall['b'].astype(str)
save_as_csv_latex_markdown(df_var_table_recall, save_dir=save_dir, name="bm25s_variants_recall")

# now let's get recall@1000 for tokenizer effect
df_tok_table_recall = df_tok.pivot_table(
    index=["stopwords", "stemmer"], columns="dataset", values="r@1000"
)

df_tok_table_recall.index.names = [index_shorthands[col] for col in df_tok_table_recall.index.names]

df_tok_table_recall["Avg."] = df_tok_table_recall.mean(axis=1).astype(float).round(1)

# Avg should be the first column
df_tok_table_recall = df_tok_table_recall[
    ["Avg."] + [col for col in df_tok_table_recall.columns if col != "Avg."]
]

df_tok_table_recall.reset_index(inplace=True)
save_as_csv_latex_markdown(df_tok_table_recall, save_dir, "tokenizer_effect_recall")