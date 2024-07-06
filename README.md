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

# for PISA
python -m benchmark.on_pisa -d "<dataset>"
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

| dataset          |   PISA  |   BM25S |    ES |   PSRN |     PT |   Rank |
|:-----------------|--------:|--------:|------:|-------:|-------:|-------:|
| arguana          |  128.01 |  573.91 | 13.67 |  11.95 | 110.51 |   2    |
| climate-fever    |   25.66 |   13.09 |  4.02 |   8.06 | OOM    |   0.03 |
| cqadupstack      |  281.65 |  170.91 | 13.38 | DNT    | OOM    |   0.77 |
| dbpedia-entity   |  114.05 |   13.44 | 10.68 |  12.69 | OOM    |   0.11 |
| fever            |   60.76 |   20.19 |  7.45 |  10.52 | OOM    |   0.06 |
| fiqa             |  546.61 |  717.78 | 16.96 |  12.51 |  20.52 |   4.46 |
| hotpotqa         |   39.70 |   20.88 |  7.11 |  10.41 | OOM    |   0.04 |
| msmarco          |  125.18 |   12.2  | 11.88 |  11.01 | OOM    |   0.07 |
| nfcorpus         | 2639.38 | 1196.16 | 45.84 |  32.94 | 256.67 | 224.66 |
| nq               |  117.34 |   41.85 | 12.16 |  11.04 | OOM    |   0.1  |
| quora            |  528.45 |  272.04 | 21.8  |  15.58 |   6.49 |   1.18 |
| scidocs          |  678.06 |  767.05 | 17.93 |  14.1  |  41.34 |   9.01 |
| scifact          | 1100.60 | 1317.12 | 20.81 |  15.02 | 184.3  |  47.6  |
| trec-covid       |  140.92 |   85.64 |  7.34 |   8.53 |   3.73 |   1.48 |
| webis-touche2020 |  201.19 |   60.59 | 13.53 |  12.36 | OOM    |   1.1  |


Notes:
* For Rank-BM25, larger datasets are ran with 1000 samples rather than the full dataset to ensure it finishes within 12h (limit for Kaggle notebooks).
* For ES and BM25S, we can set a number of threads to use. However, you might not see an improvement, in fact you might even see a decrease in throughput in the case of BM25S due to how multi-threading is implemented. Click below to see the results.

<details>
<summary>Show BM25S & ES multi-threaded (4T) performance (Q/s)</summary>

| dataset          |    PISA |   BM25S |    ES |
|:-----------------|--------:|--------:|------:|
| arguana          |  311.00 |  211    | 33.37 |
| climate-fever    |   69.89 |   22.06 |  8.13 |
| cqadupstack      |  694.82 |  248.87 | 27.76 |
| dbpedia-entity   |  294.66 |   26.18 | 15.49 |
| fever            |  166.77 |   47.03 | 14.07 |
| fiqa             | 1240.22 |  449.82 | 36.33 |
| hotpotqa         |  109.44 |   45.02 | 10.35 |
| msmarco          |  340.29 |   21.64 | 18.19 |
| nfcorpus         | 3188.22 |  784.24 | 81.07 |
| nq               |  312.09 |   77.49 | 19.18 |
| quora            | 1534.93 |  308.58 | 43.02 |
| scidocs          | 1461.20 |  614.23 | 46.36 |
| scifact          | 2620.73 |  645.88 | 50.93 |
| trec-covid       |  268.96 |  100.88 | 13.5  |
| webis-touche2020 |  297.79 |  202.39 | 26.55 |


</details>

<details>
<summary>Show normalized table wrt Rank-BM25</summary>

| dataset          |    PISA |   BM25S |     ES |   PSRN |     PT |   Rank |
|:-----------------|--------:|--------:|-------:|-------:|-------:|-------:|
| arguana          |   64.01 |  286.96 |   6.84 |   5.98 |  55.26 |      1 |
| climate-fever    |  855.33 |  436.33 | 134    | 268.67 | nan    |      1 |
| cqadupstack      |  365.78 |  221.96 |  17.38 | nan    | nan    |      1 |
| dbpedia-entity   | 1036.82 |  122.18 |  97.09 | 115.36 | nan    |      1 |
| fever            | 1012.67 |  336.5  | 124.17 | 175.33 | nan    |      1 |
| fiqa             |  122.56 |  160.94 |   3.8  |   2.8  |   4.6  |      1 |
| hotpotqa         |  992.50 |  522    | 177.75 | 260.25 | nan    |      1 |
| msmarco          | 1788.29 |  174.29 | 169.71 | 157.29 | nan    |      1 |
| nfcorpus         |   11.75 |    5.32 |   0.2  |   0.15 |   1.14 |      1 |
| nq               | 1173.40 |  418.5  | 121.6  | 110.4  | nan    |      1 |
| quora            |  447.84 |  230.54 |  18.47 |  13.2  |   5.5  |      1 |
| scidocs          |   75.26 |   85.13 |   1.99 |   1.56 |   4.59 |      1 |
| scifact          |   23.12 |   27.67 |   0.44 |   0.32 |   3.87 |      1 |
| trec-covid       |   95.22 |   57.86 |   4.96 |   5.76 |   2.52 |      1 |
| webis-touche2020 |  182.90 |   55.08 |  12.3  |  11.24 | nan    |      1 |

</details>


#### Stats

|                  | # Docs   | # Queries   | # Tokens   |
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

#### Indexing time (docs/s)

The following results follow the same setup as the queries/s benchmarks described above (single-core).

| dataset          |     PISA |    BM25S |      ES |     PSRN |      PT |     Rank |
|:-----------------|---------:|---------:|--------:|---------:|--------:|---------:|
| arguana          |  3608.75 |  4314.79 | 3591.63 |  1225.18 |  638.1  |  5021.3  |
| climate-fever    |  5474.84 |  4364.43 | 3825.89 |  6880.42 |  nan    |  7085.51 |
| cqadupstack      |  4380.73 |  4800.89 | 3725.43 |   nan    |  nan    |  5370.32 |
| dbpedia-entity   |  9532.44 |  7576.28 | 6333.82 |  8501.7  |  nan    |  9110.36 |
| fever            |  5633.20 |  4921.88 | 3879.63 |  7007.5  |  nan    |  5482.64 |
| fiqa             |  4655.99 |  5959.25 | 4035.11 |  3735.38 |  421.51 |  6455.53 |
| hotpotqa         | 10199.26 |  7420.39 | 5455.6  | 10342.5  |  nan    |  9407.9  |
| msmarco          |  9873.08 |  7480.71 | 5391.29 |  9686.07 |  nan    | 12455.9  |
| nfcorpus         |  2484.47 |  3169.4  | 1688.15 |   692.05 |  442.2  |  3579.47 |
| nq               |  6923.50 |  6083.86 | 5742.13 |  6652.33 |  nan    |  6048.85 |
| quora            | 37954.43 | 28002.4  | 8189.75 | 22818.5  | 6251.26 | 47609.2  |
| scidocs          |  3076.90 |  4107.46 | 3008.45 |  2137.64 |  312.72 |  4232.15 |
| scifact          |  2560.91 |  3253.63 | 2649.57 |   880.53 |  442.61 |  3792.84 |
| trec-covid       |  4736.72 |  4600.14 | 2966.98 |  3768.1  |  406.37 |  4672.62 |
| webis-touche2020 |  2301.53 |  2971.96 | 2484.87 |  2718.41 |  nan    |  3115.96 |

#### NDCG@10


We use the following abbreviations for the datasets:
- `AG` for arguana
- `CD` for cqadupstack
- `CF` for climate-fever
- `DB` for dbpedia-entity
- `FQ` for fiqa
- `FV` for fever
- `HP` for hotpotqa
- `MS` for msmarco
- `NF` for nfcorpus
- `NQ` for nq
- `QR` for quora
- `SD` for scidocs
- `SF` for scifact
- `TC` for trec-covid
- `WT` for webis-touche2020


|   k1 |   b | method    |   Avg. |   AG | CD   | CF   | DB   |   FQ | FV   | HP   | MS   |   NF | NQ   |   QR |   SD |   SF |   TC | WT   |
|-----:|----:|:----------|-------:|-----:|:-----|:-----|:-----|-----:|:-----|:-----|:-----|-----:|:-----|-----:|-----:|-----:|-----:|:-----|
|  0.9 | 0.4 | Lucene    |   41.1 | 40.8 | 28.2 | 16.2 | 31.9 | 23.8 | 63.8 | 62.9 | 22.8 | 31.8 | 30.5 | 78.7 | 15.0 | 67.6 | 58.9 | 44.2 |
|  1.2 | 0.75 | ATIRE     |   39.9 | 48.7 | 30.1 | 13.7 | 30.3 | 25.3 | 50.3 | 58.5 | 22.6 | 31.8 | 29.1 | 80.5 | 15.6 | 68.1 | 61.0 | 33.2 |
|  1.2 | 0.75 | BM25+     |   39.9 | 48.7 | 30.1 | 13.7 | 30.3 | 25.3 | 50.3 | 58.5 | 22.6 | 31.8 | 29.1 | 80.5 | 15.6 | 68.1 | 61.0 | 33.2 |
|  1.2 | 0.75 | BM25L     |   39.5 | 49.6 | 29.8 | 13.5 | 29.4 | 25.0 | 46.6 | 55.9 | 21.4 | 32.2 | 28.1 | 80.3 | 15.8 | 68.7 | 62.9 | 33.0 |
|  1.2 | 0.75 | Lucene    |   39.9 | 48.7 | 30.1 | 13.7 | 30.3 | 25.3 | 50.3 | 58.5 | 22.6 | 31.8 | 29.1 | 80.5 | 15.6 | 68.0 | 61.0 | 33.2 |
|  1.2 | 0.75 | Robertson |   39.9 | 49.2 | 29.9 | 13.7 | 30.3 | 25.4 | 50.3 | 58.5 | 22.6 | 31.9 | 29.2 | 80.4 | 15.5 | 68.3 | 59.0 | 33.8 |
|  1.5 | 0.75 | ES        |   42.0 | 47.7 | 29.8 | 17.8 | 31.1 | 25.3 | 62.0 | 58.6 | 22.1 | 34.4 | 31.6 | 80.6 | 16.3 | 69.0 | 68.0 | 35.4 |
|  1.5 | 0.75 | Lucene    |   39.7 | 49.3 | 29.9 | 13.6 | 29.9 | 25.1 | 48.1 | 56.9 | 21.9 | 32.1 | 28.5 | 80.4 | 15.8 | 68.7 | 62.3 | 33.1 |
|  1.5 | 0.75 | PSRN      |   40.0 | 48.4 | 29.8 | 14.2 | 30.0 | 25.3 | 50.0 | 57.6 | 22.1 | 32.6 | 28.6 | 80.6 | 15.6 | 68.8 | 63.4 | 33.5 |
|  1.5 | 0.75 | PT        |   45.0 | 44.9 | --   | --   | --   | 22.5 | --   | --   | --   | 31.9 | --   | 75.1 | 14.7 | 67.8 | 58.0 | --   |
|  1.5 | 0.75 | Rank      |   39.6 | 49.5 | 29.6 | 13.6 | 29.9 | 25.3 | 49.3 | 58.1 | 21.1 | 32.1 | 28.5 | 80.3 | 15.8 | 68.5 | 60.1 | 32.9 |

#### Recall@1000

|   k1 |   b | method    |   Avg. |   AG | CD   | CF   | DB   |   FQ | FV   | HP   | MS   |   NF | NQ   |   QR |   SD |   SF |   TC | WT   |
|-----:|----:|:----------|-------:|-----:|:-----|:-----|:-----|-----:|:-----|:-----|:-----|-----:|:-----|-----:|-----:|-----:|-----:|:-----|
|  0.9 | 0.4 | Lucene    |   77.3 | 98.8 | 71.1 | 63.3 | 67.5 | 74.3 | 95.7 | 88.0 | 85.3 | 47.7 | 89.6 | 99.5 | 56.5 | 97.0 | 39.2 | 86.0 |
|  1.2 | 0.75 | ATIRE     |   77.4 | 99.3 | 73.0 | 59.0 | 67.0 | 76.5 | 94.2 | 86.8 | 85.7 | 47.8 | 89.8 | 99.5 | 57.3 | 97.0 | 40.3 | 87.2 |
|  1.2 | 0.75 | BM25+     |   77.4 | 99.3 | 73.0 | 59.0 | 67.0 | 76.5 | 94.2 | 86.8 | 85.7 | 47.8 | 89.8 | 99.5 | 57.3 | 97.0 | 40.3 | 87.2 |
|  1.2 | 0.75 | BM25L     |   77.2 | 99.4 | 73.4 | 57.3 | 66.1 | 77.3 | 93.7 | 85.7 | 85.0 | 47.7 | 89.3 | 99.5 | 57.7 | 97.0 | 40.8 | 87.5 |
|  1.2 | 0.75 | Lucene    |   77.4 | 99.3 | 73.0 | 59.0 | 67.0 | 76.5 | 94.2 | 86.8 | 85.6 | 47.8 | 89.8 | 99.5 | 57.3 | 97.0 | 40.3 | 87.2 |
|  1.2 | 0.75 | Robertson |   77.4 | 99.3 | 73.2 | 59.1 | 66.7 | 76.8 | 94.2 | 86.8 | 85.9 | 47.5 | 89.8 | 99.5 | 57.3 | 96.7 | 40.2 | 87.4 |
|  1.5 | 0.75 | ES        |   76.9 | 99.2 | 74.2 | 58.8 | 63.6 | 76.7 | 95.9 | 85.2 | 85.1 | 39.0 | 90.8 | 99.6 | 57.9 | 98.0 | 41.3 | 88.0 |
|  1.5 | 0.75 | Lucene    |   77.2 | 99.3 | 73.3 | 57.8 | 66.3 | 77.2 | 93.8 | 86.1 | 85.2 | 47.7 | 89.5 | 99.6 | 57.5 | 97.0 | 40.6 | 87.4 |
|  1.5 | 0.75 | PSRN      |   76.7 | 99.2 | 74.2 | 58.7 | 66.2 | 76.7 | 94.2 | 86.4 | 85.1 | 37.1 | 89.4 | 99.6 | 57.4 | 97.7 | 41.1 | 87.2 |
|  1.5 | 0.75 | PT        |   73.0 | 98.3 | --   | --   | --   | 72.5 | --   | --   | --   | 51.0 | --   | 98.9 | 56.0 | 97.8 | 36.3 | --   |
|  1.5 | 0.75 | Rank      |   77.1 | 99.4 | 73.4 | 57.5 | 66.4 | 77.4 | 93.6 | 87.7 | 82.6 | 47.6 | 89.5 | 99.5 | 57.4 | 96.7 | 40.5 | 87.5 |

#### Links

* [BM25+](https://www.kaggle.com/code/xhlulu/comparing-bm25s-bm25plus)
* [BM25L](https://www.kaggle.com/code/xhlulu/comparing-bm25s-bm25l)
* [ATIRE](https://www.kaggle.com/code/xhlulu/comparing-bm25s-atire)
* [Robertson](https://www.kaggle.com/code/xhlulu/comparing-bm25s-robertson)
* [Lucene (k1=1.2, b=0.75)](https://www.kaggle.com/code/xhlulu/comparing-bm25s-lucene)
* [Lucene (k1=0.9, b=0.4)](https://www.kaggle.com/code/xhlulu/comparing-bm25s-lucene-k1-0-9-b-0-4)
* Lucene (k1=1.5, b=0.75): [DB](https://www.kaggle.com/code/xhlulu/benchmark-bm25s-dbpedia-entity), [MS](https://www.kaggle.com/code/xhlulu/benchmark-bm25s-msmarco), [FV](https://www.kaggle.com/code/xhlulu/benchmark-bm25s-fever), [CF](https://www.kaggle.com/code/xhlulu/benchmark-bm25s-climate-fever), [NQ](https://www.kaggle.com/code/xhlulu/benchmark-bm25s-nq), [HP](https://www.kaggle.com/code/xhlulu/benchmark-bm25s-hotpotqa), [Remaining](https://www.kaggle.com/code/xhlulu/benchmark-bm25s-sub-1m)
* [ES](https://www.kaggle.com/code/xhlulu/run-elasticsearch-k1-1-5-b-0-75): [FV, CF](https://www.kaggle.com/code/xhlulu/benchmark-elasticsearch-fever), [NQ](https://www.kaggle.com/code/xhlulu/benchmark-elasticsearch-nq), [MS](https://www.kaggle.com/code/xhlulu/benchmark-elasticsearch-msmarco), [HP](https://www.kaggle.com/code/xhlulu/benchmark-elasticsearch-hotpotqa), [DB](https://www.kaggle.com/code/xhlulu/benchmark-elasticsearch-dbpedia-entity), [Remaining](https://www.kaggle.com/code/xhlulu/benchmark-elasticsearch-sub-1m)
* PT: [MS](https://www.kaggle.com/code/xhlulu/benchmark-bm25-pt-msmarco), [CF](https://www.kaggle.com/code/xhlulu/benchmark-bm25-pt-climate-fever), [FV](https://www.kaggle.com/code/xhlulu/benchmark-bm25-pt-fever), [DB](https://www.kaggle.com/code/xhlulu/benchmark-bm25-pt-dbpedia-entity), [HP](https://www.kaggle.com/code/xhlulu/benchmark-bm25-pt-hotpotqa), [NQ](https://www.kaggle.com/code/xhlulu/benchmark-bm25-pt-nq), [WT](https://www.kaggle.com/code/xhlulu/benchmark-bm25-pt-webis-touche2020), [CD](https://www.kaggle.com/code/xhlulu/benchmark-bm25-pt-cqadupstack), [Remaining](https://www.kaggle.com/code/xhlulu/benchmark-bm25-pt-sub-1m)
* Rank: [DB](https://www.kaggle.com/code/xhlulu/benchmark-rank-bm25-dbpedia-entity), [HP](https://www.kaggle.com/code/xhlulu/benchmark-rank-bm25-hotpotqa), [CF](https://www.kaggle.com/code/xhlulu/benchmark-rank-bm25-climate-fever), [FV](https://www.kaggle.com/code/xhlulu/benchmark-rank-bm25-fever), [MS](https://www.kaggle.com/code/xhlulu/benchmark-rank-bm25-msmarco), [NQ](https://www.kaggle.com/code/xhlulu/benchmark-rank-bm25-nq), [CD](https://www.kaggle.com/code/xhlulu/benchmark-rank-bm25-cqadupstack), [Remaining](https://www.kaggle.com/code/xhlulu/benchmark-rank-bm25-sub-1m)
* PSRN: [CD](https://www.kaggle.com/code/xhlulu/benchmark-pyserini-cqadupstack), [FV](https://www.kaggle.com/code/xhlulu/benchmark-pyserini-fever), [HP](https://www.kaggle.com/code/xhlulu/benchmark-pyserini-hotpotqa), [MS](https://www.kaggle.com/code/xhlulu/benchmark-pyserini-msmarco), [DB](https://www.kaggle.com/code/xhlulu/benchmark-pyserini-dbpedia-entity), [NQ](https://www.kaggle.com/code/xhlulu/benchmark-pyserini-nq), [Remaining](https://www.kaggle.com/code/xhlulu/benchmark-pyserini-sub-1m)
* PISA: [NQ](https://www.kaggle.com/smac2048/pisa-nq), [DB](https://www.kaggle.com/code/smac2048/pisa-dbpedia-entity), [CF](https://www.kaggle.com/code/smac2048/pisa-climate-fever), [HP](https://www.kaggle.com/code/smac2048/pisa-hotpotqa), [FV](https://www.kaggle.com/code/smac2048/pisa-fever), [MS](https://www.kaggle.com/code/smac2048/pisa-msmarco), [CD](https://www.kaggle.com/code/smac2048/pisa-cqadupstack), [Remaining](https://www.kaggle.com/code/smac2048/pisa-rest)
