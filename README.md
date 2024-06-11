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

# For elastic, After starting the server, run:
python -m benchmark.on_elastic -d "<dataset>"
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

### Elasticsearch server

If you want to use elastic search, you need to start the server first. 

First, download the elastic search from [here](https://www.elastic.co/downloads/past-releases/elasticsearch-8-14-0). You will get a file, e.g. `elasticsearch-8.14.0-linux-x86_64.tar.gz`. Extract the file and ensure it is in the same directory as the `bm25-benchmarks` directory.

```bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.14.0-linux-x86_64.tar.gz
tar -xzf elasticsearch-8.14.0-linux-x86_64.tar.gz
# remove the tar file
rm elasticsearch-8.14.0-linux-x86_64.tar.gz
```

Then, start the server with the following command:

```bash
./elasticsearch-8.14.0/bin/elasticsearch -E xpack.security.enabled=false -E thread_pool.search.size=1 -E thread_pool.write.size=1
```

## Results

The results are benchmarked using Kaggle notebooks to ensure reproducibility. Each one is run on single-core, Intel Xeon CPU @ 2.20GHz, using 30GB RAM.

The shorthands used are:
- `BM25PT` for `bm25_pt`
- `PSRN` for `pyserini`
- `R-BM25` for `rank-bm25`
- `ES` for `elasticsearch`
- `OOM` for out-of-memory error
- `DNT` for did not terminate (i.e. went over 12 hours)

### Queries per second

| dataset          |    ES |   PSRN |     PT |   Rank |
|:-----------------|------:|-------:|-------:|-------:|
| arguana          | 13.67 |  11.95 | 110.51 |   2    |
| climate-fever    |  4.02 |   8.06 | nan    |   0.03 |
| cqadupstack      | 13.38 | nan    | nan    |   0.77 |
| dbpedia-entity   | 10.68 |  12.69 | nan    |   0.11 |
| fever            |  7.45 |  10.52 | nan    |   0.06 |
| fiqa             | 16.96 |  12.51 |  20.52 |   4.46 |
| hotpotqa         |  7.11 | nan    | nan    |   0.04 |
| msmarco          | 11.88 |  11.01 | nan    |   0.07 |
| nfcorpus         | 45.84 |  32.94 | 256.67 | 224.66 |
| nq               | 12.16 |  11.04 | nan    |   0.1  |
| quora            | 21.8  |  15.58 |   6.49 |   1.18 |
| scidocs          | 17.93 |  14.1  |  41.34 |   9.01 |
| scifact          | 20.81 |  15.02 | 184.3  |  47.6  |
| trec-covid       |  7.34 |   8.53 |   3.73 |   1.48 |
| webis-touche2020 | 13.53 |  12.36 | nan    |   1.1  |

#### Stats

|                  | num_docs   | num_queries   | num_tokens   |
|:-----------------|:-----------|:--------------|:-------------|
| msmarco          | 8,841,823  | 6,980         | 340,859,891  |
| hotpotqa         | 5,233,329  | 7,405         | 169,530,287  |
| trec-covid       | 171,332    | 50            | 20,231,412   |
| webis-touche2020 | 382,545    | 49            | 74,180,340   |
| arguana          | 8,674      | 1,406         | 947,470      |
| fiqa             | 57,638     | 648           | 5,189,035    |
| nfcorpus         | 3,633      | 323           | 614,081      |
| climate-fever    | 5,416,593  | 1,535         | 318,190,120  |
| nq               | 2,681,468  | 3,452         | 148,249,808  |
| scidocs          | 25,657     | 1,000         | 3,211,248    |
| quora            | 522,931    | 10,000        | 4,202,123    |
| dbpedia-entity   | 4,635,922  | 400           | 162,336,256  |
| cqadupstack      | 457,199    | 13,145        | 44,857,487   |
| fever            | 5,416,568  | 6,666         | 318,184,321  |
| scifact          | 5,183      | 300           | 812,074      |

#### NDCG@10

| dataset          |     ES |     PSRN |       PT |   Rank |
|:-----------------|-------:|---------:|---------:|-------:|
| arguana          | 0.4716 |   0.4845 |   0.449  | 0.4946 |
| climate-fever    | 0.1862 |   0.1416 | nan      | 0.1358 |
| cqadupstack      | 0.301  | nan      | nan      | 0.2961 |
| dbpedia-entity   | 0.3202 |   0.2997 | nan      | 0.2989 |
| fever            | 0.6494 |   0.5004 | nan      | 0.4934 |
| fiqa             | 0.2536 |   0.2531 |   0.2252 | 0.2528 |
| hotpotqa         | 0.6022 | nan      | nan      | 0.5813 |
| msmarco          | 0.2275 |   0.221  | nan      | 0.2109 |
| nfcorpus         | 0.3428 |   0.3259 |   0.3194 | 0.3211 |
| nq               | 0.3261 |   0.2862 | nan      | 0.2851 |
| quora            | 0.8077 |   0.8063 |   0.751  | 0.8027 |
| scidocs          | 0.1647 |   0.1564 |   0.1473 | 0.1581 |
| scifact          | 0.6906 |   0.6876 |   0.678  | 0.6849 |
| trec-covid       | 0.6158 |   0.6337 |   0.5804 | 0.6008 |
| webis-touche2020 | 0.3471 |   0.3346 | nan      | 0.3287 |

#### Recall@1000

| dataset          |     ES |     PSRN |       PT |   Rank |
|:-----------------|-------:|---------:|---------:|-------:|
| arguana          | 0.9915 |   0.9922 |   0.9829 | 0.9936 |
| climate-fever    | 0.5977 |   0.5868 | nan      | 0.5751 |
| cqadupstack      | 0.744  | nan      | nan      | 0.7338 |
| dbpedia-entity   | 0.6387 |   0.6621 | nan      | 0.6637 |
| fever            | 0.9598 |   0.942  | nan      | 0.9364 |
| fiqa             | 0.7591 |   0.7671 |   0.7246 | 0.7737 |
| hotpotqa         | 0.8577 | nan      | nan      | 0.877  |
| msmarco          | 0.8548 |   0.8513 | nan      | 0.8258 |
| nfcorpus         | 0.39   |   0.3715 |   0.5101 | 0.4762 |
| nq               | 0.9107 |   0.8945 | nan      | 0.8954 |
| quora            | 0.9958 |   0.9958 |   0.9895 | 0.9954 |
| scidocs          | 0.5785 |   0.5736 |   0.5599 | 0.5745 |
| scifact          | 0.98   |   0.9767 |   0.9783 | 0.9667 |
| trec-covid       | 0.362  |   0.411  |   0.3627 | 0.4055 |
| webis-touche2020 | 0.882  |   0.8722 | nan      | 0.8749 |
