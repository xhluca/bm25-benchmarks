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

## Results

The results are benchmarked using Kaggle notebooks to ensure reproducibility. Each one is run on single-core, Intel Xeon CPU @ 2.20GHz, using 30GB RAM.

The shorthands used are:
- `BM25PT` for `bm25_pt`
- `PSRN` for `pyserini`
- `R-BM25` for `rank-bm25`
- `OOM` for out-of-memory error
- `DNT` for did not terminate (i.e. went over 12 hours)

### Queries per second

| dataset          |   BM25PT |   PSRN |   R-BM25 |
|:-----------------|---------:|-------:|---------:|
| arguana          |   110.51 |  11.48 |     2    |
| climate-fever    |   OOM    |   8    |   DNT    |
| cqadupstack      |   OOM    | DNT    |     0.77 |
| dbpedia-entity   |   OOM    |  11.57 |   DNT    |
| fever            |   OOM    |  10.82 |   DNT    |
| fiqa             |    20.52 |  11.56 |     4.46 |
| hotpotqa         |   OOM    |   9.58 |   DNT    |
| msmarco          |   OOM    |  11.24 |   DNT    |
| nfcorpus         |   256.67 |  32.66 |   224.66 |
| nq               |   OOM    |  11.25 |   DNT    |
| quora            |     6.49 |  15.25 |     1.18 |
| scidocs          |    41.34 |  14.45 |     9.01 |
| scifact          |   184.3  |  14.89 |    47.6  |
| trec-covid       |     3.73 |   8.6  |     1.48 |
| webis-touche2020 |   OOM    |  12.22 |     1.1  |
