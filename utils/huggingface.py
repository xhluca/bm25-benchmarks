
def batch_tokenize(tokenizer, texts, add_special_tokens=False, verbose=False):
    from tqdm.auto import tqdm

    tokenizer_kwargs = dict(
        return_attention_mask=False,
        return_token_type_ids=False,
        add_special_tokens=add_special_tokens,
        max_length=None,
    )
    tokenized = tokenizer(texts, **tokenizer_kwargs)
    output = []

    for i in tqdm(range(len(texts)), desc="Processing tokens (huggingface tokenizer)", leave=False, disable=not verbose):
        output.append(tokenized[i].tokens)

    return output

