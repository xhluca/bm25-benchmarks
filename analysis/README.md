# Analysis

## Download results from Kaggle

The first step is to download the results from Kaggle. To do that, we need to use the API:

```bash
pip install kaggle
```

We need to input the username and API key as environment variables (can be generated [in the settings](https://www.kaggle.com/settings)):
```bash
export KAGGLE_USERNAME=<username>
export KAGGLE_KEY=<key>
```

Then we can download the results:

```bash
python analysis/download_results.py
```

Now, to combine results from all the runs, we can use the following command:

```bash
python analysis/combine_results.py
```

You can find them in `analysis/out/`.