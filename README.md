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
|   k1 |    b | method    |    Avg. |      AG | CD                 | CF                 | DB   |   FQ | FV                 | HP   | MS                 |      NF | NQ   |   QR |   SD |   SF |      TC | WT   |
|-----:|-----:|:----------|--------:|--------:|:-------------------|:-------------------|:-----|-----:|:-------------------|:-----|:-------------------|--------:|:-----|-----:|-----:|-----:|--------:|:-----|
|  0.9 | 0.4  | Lucene    | 41.14   | 40.8    | 28.2               | 16.2               | 31.9 | 23.8 | 63.8               | 62.9 | 22.8               | 31.8    | 30.5 | 78.7 | 15   | 67.6 | 58.9    | 44.2 |
|  1.2 | 0.75 | ATIRE     | 39.92   | 48.7    | 30.1               | 13.7               | 30.3 | 25.3 | 50.3               | 58.5 | 22.6               | 31.8    | 29.1 | 80.5 | 15.6 | 68.1 | 61      | 33.2 |
|  1.2 | 0.75 | BM25+     | 39.92   | 48.7    | 30.1               | 13.7               | 30.3 | 25.3 | 50.3               | 58.5 | 22.6               | 31.8    | 29.1 | 80.5 | 15.6 | 68.1 | 61      | 33.2 |
|  1.2 | 0.75 | BM25L     | 39.48   | 49.6    | 29.8               | 13.5               | 29.4 | 25   | 46.6               | 55.9 | 21.4               | 32.2    | 28.1 | 80.3 | 15.8 | 68.7 | 62.9    | 33.0 |
|  1.2 | 0.75 | Lucene    | 39.9133 | 48.7    | 30.1               | 13.7               | 30.3 | 25.3 | 50.3               | 58.5 | 22.6               | 31.8    | 29.1 | 80.5 | 15.6 | 68   | 61      | 33.2 |
|  1.2 | 0.75 | Robertson | 39.8667 | 49.2    | 29.9               | 13.7               | 30.3 | 25.4 | 50.3               | 58.5 | 22.6               | 31.9    | 29.2 | 80.4 | 15.5 | 68.3 | 59      | 33.8 |
|  1.5 | 0.75 | ES        | 41.98   | 47.7    | 29.8               | 17.8               | 31.1 | 25.3 | 62.0               | 58.6 | 22.1               | 34.4    | 31.6 | 80.6 | 16.3 | 69   | 68      | 35.4 |
|  1.5 | 0.75 | Lucene    | 39.2489 | 48.4333 | 29.8 | 13.5 | 28.8 | 24.5 | 48.3 | 56.9 | 21.6 | 31.5667 | 28.3 | 78.3 | 15.6 | 67.8 | 61.1333 | 34.1 |
|  1.5 | 0.75 | PSRN      | 40.0538 | 48.4    | --                 | 14.2               | 30.0 | 25.3 | --                 | 57.6 | 22.1               | 32.6    | 28.6 | 80.6 | 15.6 | 68.8 | 63.4    | 33.5 |
|  1.5 | 0.75 | PT        | 44.9857 | 44.9    | --                 | --                 | --   | 22.5 | --                 | --   | --                 | 31.9    | --   | 75.1 | 14.7 | 67.8 | 58      | --   |
|  1.5 | 0.75 | Rank      | 39.64   | 49.5    | 29.6               | 13.6               | 29.9 | 25.3 | 49.3               | 58.1 | 21.1               | 32.1    | 28.5 | 80.3 | 15.8 | 68.5 | 60.1    | 32.9 |

#### Recall@1000

|   k1 |   b | method    |   Avg. |   AG | CD                | CF                | DB   |   FQ | FV                | HP                | MS                |   NF | NQ                |   QR |   SD |   SF |   TC | WT   |
|-----:|----:|:----------|-------:|-----:|:------------------|:------------------|:-----|-----:|:------------------|:------------------|:------------------|-----:|:------------------|-----:|-----:|-----:|-----:|:-----|
|  0.9 | 0.4 | Lucene    |   77.3 | 98.8 | 71.1              | 63.3              | 67.5 | 74.3 | 95.7              | 88.0              | 85.3              | 47.7 | 89.6              | 99.5 | 56.5 | 97.0 | 39.2 | 86.0 |
|  1.2 | 0.75 | ATIRE     |   77.4 | 99.3 | 73.0              | 59.0              | 67.0 | 76.5 | 94.2              | 86.8              | 85.7              | 47.8 | 89.8              | 99.5 | 57.3 | 97.0 | 40.3 | 87.2 |
|  1.2 | 0.75 | BM25+     |   77.4 | 99.3 | 73.0              | 59.0              | 67.0 | 76.5 | 94.2              | 86.8              | 85.7              | 47.8 | 89.8              | 99.5 | 57.3 | 97.0 | 40.3 | 87.2 |
|  1.2 | 0.75 | BM25L     |   77.2 | 99.4 | 73.4              | 57.3              | 66.1 | 77.3 | 93.7              | 85.7              | 85.0              | 47.7 | 89.3              | 99.5 | 57.7 | 97.0 | 40.75 | 87.5 |
|  1.2 | 0.75 | Lucene    |   77.4 | 99.3 | 73.0              | 59.0              | 67.0 | 76.5 | 94.2              | 86.8              | 85.6              | 47.8 | 89.8              | 99.5 | 57.3 | 97.0 | 40.3 | 87.2 |
|  1.2 | 0.75 | Robertson |   77.4 | 99.3 | 73.2              | 59.1              | 66.7 | 76.8 | 94.2              | 86.8              | 85.9              | 47.5 | 89.8              | 99.5 | 57.3 | 96.7 | 40.2 | 87.4 |
|  1.5 | 0.75 | ES        |   76.9 | 99.2 | 74.2              | 58.8              | 63.6 | 76.7 | 95.9              | 85.2              | 85.1              | 39.0 | 90.8              | 99.6 | 57.9 | 98.0 | 41.3 | 88.0 |
|  1.5 | 0.75 | Lucene    |   76.5 | 99.2 | 72.2 | 57.5 | 64.0 | 75.6 | 93.9 | 86.0 | 83.87 | 47.3 | 89.0 | 99.3 | 57.1 | 96.6 | 39.1 | 86.8 |
|  1.5 | 0.75 | PSRN      |   75.5 | 99.2 | --                | 58.7              | 66.2 | 76.7 | --                | 86.4              | 85.1              | 37.1 | 89.4              | 99.6 | 57.4 | 97.7 | 41.1 | 87.2 |
|  1.5 | 0.75 | PT        |   73.0 | 98.3 | --                | --                | --   | 72.5 | --                | --                | --                | 51.0 | --                | 98.9 | 56.0 | 97.8 | 36.3 | --   |
|  1.5 | 0.75 | Rank      |   77.1 | 99.4 | 73.4              | 57.5              | 66.4 | 77.4 | 93.6              | 87.7              | 82.6              | 47.6 | 89.5              | 99.5 | 57.4 | 96.7 | 40.5 | 87.5 |
