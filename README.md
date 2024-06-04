# BM25 Benchmarks

## Benchmarking

To run benchmark on bm25 implementations, simply run:

```bash
# For bm25_pt
python -m benchmark.on_bm25_pt -d "<dataset>"

# For rank-bm25
python -m benchmark.on_rank_bm25 -d "<dataset>"

# for Pyserini
python -m benchmark.on_pyserini -d "<dataset>"
```

where `<dataset>` is the name of the dataset to be used. 

### Available datasets

The available datasets are public BEIR datasets:
- `trec-covid`
- `nfcorpus`
- `fiqa`
- `arguana`
- `webis-touche2020`
- `quora`
- `scidocs`
- `scifact`
- `cqadupstack`
- `nq`
- `msmarco`
- `hotpotqa`
- `dbpedia-entity`
- `fever`
- `climate-fever`

### Sampling during benchmarking

For `rank-bm25`, due to the long runtime, we can sample queries
```bash
python -m benchmark.on_rank_bm25 -d "<dataset>" --samples <num_samples>
```

### Rank-bm25 variants

For `rank-bm25`, we can also specify the method with `--method` to be used:
- `rank` (default)
- `bm25l`
- `bm25+`

Results will be saved in `results/` directory.