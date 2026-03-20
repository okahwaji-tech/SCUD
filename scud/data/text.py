"""Text data loading: tokenizers, dataset processing, dataloaders."""

from __future__ import annotations

import functools
import itertools
import json
import os
import re
import shutil
import typing
import urllib
import zipfile

import torch

from scud.data._helpers import fsspec_exists, fsspec_mkdirs, get_logger

_HAS_TEXT_DEPS = True
try:
    import datasets
    import fsspec as _fsspec
    import requests
    import tokenizers
    import transformers
except ImportError:
    _HAS_TEXT_DEPS = False

LOGGER = get_logger(__name__)

text_data_name_dict: dict[str, None] = {"lm1b": None}


# ---------------------------------------------------------------------------
# Detokenizers
# ---------------------------------------------------------------------------


def wt_detokenizer(string: str) -> str:
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def ptb_detokenizer(x: str) -> str:
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x


def lm1b_detokenizer(x: str) -> str:
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(r" \'(\w+)", r"'\1", x)
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("\u00a3 ", "\u00a3")
    return x


def lambada_detokenizer(text: str) -> str:
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    return "\n" + text.strip()


def scientific_papers_detokenizer(x: str) -> str:
    x = wt_detokenizer(x)
    x = lm1b_detokenizer(x)
    return x


# ---------------------------------------------------------------------------
# Text8 tokenizer
# ---------------------------------------------------------------------------


class Text8Tokenizer(transformers.PreTrainedTokenizer):
    def __init__(
        self,
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
        unk_token: str = "[UNK]",
        **kwargs: object,
    ):
        self.characters = list("abcdefghijklmnopqrstuvwxyz ")
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[MASK]": 4,
            "[PAD]": 5,
            "[RESERVED]": 6,
            "[UNK]": 7,
            **{ch: i + 8 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs: object) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def get_lambada_test_dataset() -> "datasets.Dataset":
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url: str) -> list[dict[str, str]]:
        response = requests.get(url, stream=True)
        data_list: list[dict[str, str]] = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset


def get_text8_dataset(
    cache_dir: str,
    max_seq_length: int = 256,
    drop_last: bool = True,
    crop_train: bool = False,
) -> "datasets.DatasetDict":
    """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.
      max_seq_length: int, maximum length of sequences.
          (default: 256, as in D3PM codebase.)
      drop_last: bool, whether to drop the last incomplete
          batch. (default: True, as in D3PM codebase.)
      crop_train: bool, whether to subsample contiguous
          subsequences from training example. serves to
          make sure transformer models with absolute position
          embeddings do not have incorrect position-wise
          marginals. (default: False, but necessary to match D3PM AR)

    Returns:
      dataset: dataset.DatasetDict, with keys 'train',
          'valid', 'test'.
    """
    url = "http://mattmahoney.net/dc/text8.zip"
    if not crop_train:
        cache_dir = f"{cache_dir}/text8"
    else:
        cache_dir = f"{cache_dir}/text8-crop-train"
    split_names = ["train", "validation", "test"]
    if not all(
        [fsspec_exists(os.path.join(cache_dir, split)) for split in split_names]
    ):
        # Check if raw data exists
        raw_cache_dir = os.path.join(cache_dir, "raw_data")
        if not all(
            [
                fsspec_exists(os.path.join(raw_cache_dir, f"text8.{split}.txt"))
                for split in split_names
            ]
        ):
            if not fsspec_exists(os.path.join(raw_cache_dir, "text8.zip")):
                fsspec_mkdirs(raw_cache_dir, exist_ok=True)
                LOGGER.info("Downloading text8 from URL {}.".format(url))
                with urllib.request.urlopen(url) as in_stream:
                    with open(
                        os.path.join(raw_cache_dir, "text8.zip"), "wb"
                    ) as out_file:
                        shutil.copyfileobj(in_stream, out_file)

            with _fsspec.open(os.path.join(raw_cache_dir, "text8.zip"), "rb") as f:
                rawdata = zipfile.ZipFile(f).read("text8").decode("utf-8")

            # Splits taken from D3PM codebase
            splits = {
                "train": rawdata[:90000000],
                "validation": rawdata[90000000:95000000],
                "test": rawdata[95000000:],
            }

            for split, data in splits.items():
                _path = os.path.join(raw_cache_dir, f"text8.{split}.txt")
                with _fsspec.open(_path, "w") as f:
                    f.write(data)
        else:
            splits = {}
            for split in split_names:
                _path = os.path.join(raw_cache_dir, f"text8.{split}.txt")
                with _fsspec.open(_path, "r") as f:
                    splits[split] = f.read()

        # Chunk and save as datasets.DatasetDict
        def chunks(lst: str, n: int):  # noqa: ANN202
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        dataset_dict = {}
        for k, v in splits.items():
            if k == "train" and crop_train is True:
                chunk_size = 2 * max_seq_length
            else:
                chunk_size = max_seq_length
            text = list(chunks(v, chunk_size))
            if drop_last and len(text[-1]) < chunk_size:
                text = text[:-1]
            dataset_dict[k] = datasets.Dataset.from_dict({"text": text})
        dataset = datasets.DatasetDict(dataset_dict)
        dataset.save_to_disk(cache_dir)
    else:
        dataset = datasets.load_from_disk(cache_dir)

    return dataset


def _group_texts(
    examples: dict[str, list[list[int]]],
    block_size: int,
    bos: int,
    eos: int,
) -> dict[str, list[list[int] | torch.Tensor]]:
    # Concatenate all texts.
    concatenated_examples = list(itertools.chain(*examples["input_ids"]))
    total_length = len(concatenated_examples)
    # TODO(yair): look into not dropping the remainder but rather padding it.
    # We drop the small remainder, and if the total_length < block_size - 2
    # we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of
    # this drop, you can customize this part to your needs.
    new_block_size = block_size - 2  # [BOS] and [EOS] to be added
    total_length = (total_length // new_block_size) * new_block_size
    # Split by chunks of max_len.
    result: dict[str, list[list[int] | torch.Tensor]] = {}
    _values: list[list[int]] = []
    _attn_masks: list[torch.Tensor] = []
    for i in range(0, total_length, new_block_size):
        _values.append(
            [bos] + concatenated_examples[i : i + new_block_size] + [eos]
        )
        _attn_masks.append(torch.ones(block_size))
    result["input_ids"] = _values
    result["attention_mask"] = _attn_masks
    return result


def get_dataset(
    dataset_name: str,
    tokenizer: "transformers.PreTrainedTokenizer",
    wrap: bool,
    mode: str,
    cache_dir: str,
    block_size: int = 1024,
    num_proc: int = len(os.sched_getaffinity(0)),
    streaming: bool = False,
) -> "datasets.Dataset":
    if wrap:
        filename = f"{dataset_name}_{mode}_bs{block_size}_wrapped.dat"
    else:
        filename = f"{dataset_name}_{mode}_bs{block_size}_unwrapped.dat"
    _path = os.path.join(cache_dir, filename)

    if fsspec_exists(_path):
        LOGGER.info(f"Loading data from: {_path}")
        return datasets.load_from_disk(_path).with_format("torch")
    LOGGER.info(f"Generating new data at: {_path}")

    crop_train = dataset_name == "text8-crop"
    if mode == "train" and crop_train:
        # double block size for sub-sampling
        block_size *= 2

    if dataset_name == "wikitext103":
        dataset = datasets.load_dataset(
            "wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir
        )
    elif dataset_name == "wikitext2":
        dataset = datasets.load_dataset(
            "wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir
        )
    elif dataset_name == "ptb":
        dataset = datasets.load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif dataset_name == "lambada":
        dataset = get_lambada_test_dataset()
    elif dataset_name == "text8":
        assert wrap
        dataset = get_text8_dataset(cache_dir, max_seq_length=block_size)
    elif dataset_name == "text8-crop":
        dataset = get_text8_dataset(
            cache_dir, max_seq_length=block_size, crop_train=True
        )
    elif dataset_name == "openwebtext-train":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[:-100000]",
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True,
        )
    elif dataset_name == "openwebtext-valid":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[-100000:]",
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True,
        )
    elif dataset_name == "scientific_papers_arxiv":
        dataset = datasets.load_dataset(
            "scientific_papers",
            "arxiv",
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming,
        )
    elif dataset_name == "scientific_papers_pubmed":
        dataset = datasets.load_dataset(
            "scientific_papers",
            "pubmed",
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming,
        )
    elif dataset_name == "ag_news":
        dataset = datasets.load_dataset(
            "ag_news",
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True,
        )
    else:
        dataset = datasets.load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True,
        )

    if dataset_name in ["lambada", "openwebtext-train", "openwebtext-valid"]:
        data = dataset
    else:
        data = dataset[mode]

    if dataset_name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif dataset_name == "ptb":
        detokenizer = ptb_detokenizer
    elif dataset_name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif dataset_name == "lambada":
        detokenizer = lambada_detokenizer
    elif dataset_name.startswith("scientific_papers"):
        detokenizer = scientific_papers_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detok_fn):  # noqa: ANN001, ANN202
        def detok(text):  # noqa: ANN001, ANN202
            for i, t in enumerate(text, 0):
                text[i] = detok_fn(t)
            return text

        return detok

    EOS = tokenizer.encode(tokenizer.eos_token)[0]
    BOS = tokenizer.encode(tokenizer.bos_token)[0]

    def preprocess_and_tokenize(example):  # noqa: ANN001, ANN202
        if dataset_name == "ptb":
            text = example["sentence"]
        elif "scientific_papers" in dataset_name:
            text = example["article"]
        else:
            text = example["text"]

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"

        if wrap:
            tokens = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            tokens = {"input_ids": [t + [EOS] for t in tokens["input_ids"]]}
            # Still missing BOS, but will be added in group_texts
        else:
            tokens = tokenizer(
                text,
                max_length=block_size,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
        return tokens

    if streaming:
        tokenized_dataset = data.map(
            preprocess_and_tokenize, batched=True, desc="Tokenizing"
        )
    else:
        tokenized_dataset = data.map(
            preprocess_and_tokenize,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc="Tokenizing",
        )
    if dataset_name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns("sentence")
    elif "scientific_papers" in dataset_name:
        tokenized_dataset = tokenized_dataset.remove_columns(
            ["article", "abstract", "section_names"]
        )
    elif dataset_name == "ag_news":
        tokenized_dataset = tokenized_dataset.remove_columns(["text", "label"])
    else:
        tokenized_dataset = tokenized_dataset.remove_columns("text")

    if not wrap:
        tokenized_dataset.save_to_disk(_path)
        return tokenized_dataset.with_format("torch")

    group_texts = functools.partial(
        _group_texts, block_size=block_size, bos=BOS, eos=EOS
    )
    if streaming:
        chunked_dataset = tokenized_dataset.map(
            group_texts, batched=True, desc="Grouping"
        )
    else:
        chunked_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc="Grouping",
        )
        chunked_dataset.save_to_disk(_path)
    chunked_dataset = chunked_dataset.with_format("torch")
    return chunked_dataset


# ---------------------------------------------------------------------------
# Tokenizer setup
# ---------------------------------------------------------------------------


def get_tokenizer(config: object) -> "transformers.PreTrainedTokenizer":
    if config.data.tokenizer_name_or_path == "text8":
        tokenizer = Text8Tokenizer()
    elif config.data.tokenizer_name_or_path == "bert-base-uncased":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif config.data.tokenizer_name_or_path == "gpt2":
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.data.tokenizer_name_or_path
        )

    if isinstance(tokenizer, transformers.GPT2TokenizerFast) or isinstance(
        tokenizer, transformers.GPT2Tokenizer
    ):
        tokenizer._tokenizer.post_processor = (
            tokenizers.processors.BertProcessing(
                (tokenizer.bos_token, tokenizer.bos_token_id),
                (tokenizer.eos_token, tokenizer.eos_token_id),
            )
        )

    # For wrapped batches:
    #  [BOS] sent1 [EOS] sent2-fragment [EOS]
    #  [BOS] sent2-fragment [EOS] sent3 [EOS]
    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(
                "Tokenizer must have a bos_token or "
                f"cls_token: {tokenizer}"
            )
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(
                "Tokenizer must have a eos_token "
                f"or sep_token: {tokenizer}"
            )
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


# ---------------------------------------------------------------------------
# Main text dataloader builder
# ---------------------------------------------------------------------------


def get_text_dataloaders(
    config: object,
    tokenizer: "transformers.PreTrainedTokenizer",
    skip_train: bool = False,
    skip_valid: bool = False,
    valid_seed: int | None = None,
) -> tuple[torch.utils.data.DataLoader | None, torch.utils.data.DataLoader | None]:
    if skip_train:
        train_set = None
    else:
        train_set = get_dataset(
            config.data.data,
            tokenizer,
            mode="train",
            wrap=config.data.wrap,
            cache_dir=config.data.cache_dir,
            # block_size=config.model.length
        )

    if config.data.data in ["text8", "lm1b", "ag_news"]:
        validation_split = "test"
    else:
        validation_split = "validation"
    if skip_valid:
        valid_set = None
    else:
        valid_set = get_dataset(
            config.data.valid,
            tokenizer,
            wrap=config.data.wrap,
            mode=validation_split,
            cache_dir=config.data.cache_dir,
            #   block_size=config.model.length,
            streaming=False,
        )

    # multiprocessing.cpu_count()
    num_workers = 16 // max([1, torch.cuda.device_count()])

    if skip_train:
        train_loader = None
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.train.batch_size,
            num_workers=num_workers,
            #   pin_memory=config.loader.pin_memory,
            #   shuffle=not config.data.streaming,
            persistent_workers=True,
        )
        train_loader.tokenizer = tokenizer
    if skip_valid:
        valid_loader = None
    else:
        if valid_seed is None:
            shuffle_valid = False
            generator = None
        else:
            shuffle_valid = True
            generator = torch.Generator().manual_seed(valid_seed)
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.train.batch_size,
            num_workers=num_workers,
            #   pin_memory=config.loader.pin_memory,
            shuffle=shuffle_valid,
            generator=generator,
        )
        # Will be used in generative perplexity calculation
        valid_loader.tokenizer = tokenizer

    return train_loader, valid_loader
