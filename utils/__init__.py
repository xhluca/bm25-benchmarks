from functools import partial
import re
from typing import Any, Dict, List, Union, Callable, NamedTuple

class Tokenized(NamedTuple):
    ids: List[List[int]]
    vocab: Dict[str, int]

def _infer_stopwords(stopwords: Union[str, List[str]]) -> List[str]:
    STOPWORDS_EN = (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    )

    if stopwords in ["english", "en", True]:
        return STOPWORDS_EN
    elif isinstance(stopwords, str):
        raise ValueError(
            f"{stopwords} not recognized. Only default English stopwords are currently supported. "
            "Please input a list of stopwords"
        )
    else:
        return stopwords

def tokenize(
    texts,
    lower: bool = True,
    token_pattern: str = r"(?u)\b\w\w+\b",
    stopwords: Union[str, List[str]] = None,
    stemmer: Callable = None,
    return_ids: bool = False,
    leave: bool = False,
    verbose: bool = False,
):
    from tqdm.auto import tqdm
    if isinstance(texts, str):
        texts = [texts]
    
    token_pattern = re.compile(token_pattern)
    stopwords = _infer_stopwords(stopwords)

    # Step 1: Split the strings using the regex pattern
    split_fn = token_pattern.findall

    corpus_ids = []
    token_to_index = {}

    tqdm = partial(tqdm, disable=not verbose)
    for text in tqdm(texts, desc="Split strings", leave=leave):
        stopwords_set = set(stopwords)
        if lower:
            text = text.lower()
        
        splitted = split_fn(text)
        doc_ids = []

        for token in splitted:
            if token in stopwords_set:
                continue

            if token not in token_to_index:
                token_to_index[token] = len(token_to_index)
            
            token_id = token_to_index[token]
            doc_ids.append(token_id)
        
        corpus_ids.append(doc_ids)

    # Create a list of unique tokens that we will use to create the vocabulary
    unique_tokens = list(token_to_index.keys())

    # Step 2: Stem the tokens if a stemmer is provided
    if stemmer is not None:
        if hasattr(stemmer, "stemWords"):
            stemmer_fn = stemmer.stemWords
        elif callable(stemmer):
            stemmer_fn = stemmer
        else:
            error_msg = "Stemmer must have a `stemWord` method, or be callable. For example, you can use the PyStemmer library."
            raise ValueError(error_msg)

        # Now, we use the stemmer on the token_to_index dictionary to get the stemmed tokens
        tokens_stemmed = stemmer_fn(unique_tokens)
        vocab = set(tokens_stemmed)
        vocab_dict = {token: i for i, token in enumerate(vocab)}
        stem_id_to_stem = {v: k for k, v in vocab_dict.items()}
        # We create a dictionary mapping the stemmed tokens to their index
        doc_id_to_stem_id = {
            token_to_index[token]: vocab_dict[stem]
            for token, stem in zip(unique_tokens, tokens_stemmed)
        }

        # Now, we simply need to replace the tokens in the corpus with the stemmed tokens
        for i, doc_ids in enumerate(tqdm(corpus_ids, desc="Stem Tokens", leave=leave)):
            corpus_ids[i] = [doc_id_to_stem_id[doc_id] for doc_id in doc_ids]
    else:
        vocab_dict = token_to_index
    
    # Step 3: Return the tokenized IDs and the vocab dictionary or the tokenized strings
    if return_ids:
        return Tokenized(ids=corpus_ids, vocab=vocab_dict)

    else:
        # We need a reverse dictionary to convert the token IDs back to tokens
        reverse_dict = stem_id_to_stem if stemmer is not None else unique_tokens
        # We convert the token IDs back to tokens in-place
        for i, token_ids in enumerate(
            tqdm(corpus_ids, desc="Reconstructing token strings", leave=leave)
        ):
            corpus_ids[i] = [reverse_dict[token_id] for token_id in token_ids]

        return corpus_ids
