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
| climate-fever    |   OOM    |   8    |     0.03 |
| cqadupstack      |   OOM    | OOM    |     0.77 |
| dbpedia-entity   |   OOM    |  11.57 |   DNT    |
| fever            |   OOM    |  10.82 |     0.06 |
| fiqa             |    20.52 |  11.56 |     4.46 |
| hotpotqa         |   OOM    |   9.58 |     0.04 |
| msmarco          |   OOM    |  11.24 |     0.07 |
| nfcorpus         |   256.67 |  32.66 |   224.66 |
| nq               |   OOM    |  11.25 |     0.1  |
| quora            |     6.49 |  15.25 |     1.18 |
| scidocs          |    41.34 |  14.45 |     9.01 |
| scifact          |   184.3  |  14.89 |    47.6  |
| trec-covid       |     3.73 |   8.6  |     1.48 |
| webis-touche2020 |   OOM    |  12.22 |     1.1  |

#### Stats

|                  | num_docs   | num_queries   | num_tokens   |
|:-----------------|:-----------|:--------------|:-------------|
| nfcorpus         | 3,633      | 323           | 614,081      |
| webis-touche2020 | 382,545    | 49            | 74,180,340   |
| dbpedia-entity   | 4,635,922  | 400           | 162,336,256  |
| scidocs          | 25,657     | 1,000         | 3,211,248    |
| fiqa             | 57,638     | 648           | 5,189,035    |
| scifact          | 5,183      | 300           | 812,074      |
| trec-covid       | 171,332    | 50            | 20,231,412   |
| arguana          | 8,674      | 1,406         | 947,470      |
| quora            | 522,931    | 10,000        | 4,202,123    |
| fever            | 5,416,568  | 6,666         | 318,184,321  |
| cqadupstack      | 457,199    | 13,145        | 44,857,487   |
| msmarco          | 8,841,823  | 6,980         | 340,859,891  |
| climate-fever    | 5,416,593  | 1,535         | 318,190,120  |
| nq               | 2,681,468  | 3,452         | 148,249,808  |
| hotpotqa         | 5,233,329  | 7,405         | 169,530,287  |