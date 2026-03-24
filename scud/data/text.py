"""Text data loading via HuggingFace datasets: LM1B, text8, and generic text."""

from __future__ import annotations

import functools
import itertools
import os

import torch
from omegaconf import DictConfig

from scud.data._helpers import get_logger

_HAS_TEXT_DEPS = True
try:
    import datasets
    import transformers
except ImportError:
    _HAS_TEXT_DEPS = False

LOGGER = get_logger(__name__)

HF_LM1B = "dvruette/lm1b"

# Used by loader.py to route cfg.data.data -> text pipeline.
text_data_name_dict: dict[str, str | None] = {"lm1b": HF_LM1B, "text8": None}


class Text8Tokenizer(transformers.PreTrainedTokenizer):
    """Character-level tokenizer for the text8 dataset (vocab_size=35)."""

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

    def _tokenize(self, text: str, **kwargs: object) -> list[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab_str_to_int


def get_tokenizer(config: DictConfig) -> transformers.PreTrainedTokenizer:
    """Return the tokenizer specified by ``config.data.tokenizer_name_or_path``."""
    name: str = config.data.tokenizer_name_or_path
    if name == "text8":
        tokenizer = Text8Tokenizer()
    elif name == "bert-base-uncased":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif name == "gpt2":
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)

    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(f"Tokenizer must have a bos_token or cls_token: {tokenizer}")
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(f"Tokenizer must have an eos_token or sep_token: {tokenizer}")
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def _group_texts(
    examples: dict[str, list[list[int]]],
    block_size: int,
    bos: int,
    eos: int,
) -> dict[str, list[object]]:
    """Concatenate tokenised texts and rechunk into ``[BOS] ... [EOS]`` blocks."""
    concatenated = list(itertools.chain(*examples["input_ids"]))
    usable = block_size - 2  # room for [BOS] and [EOS]
    total = (len(concatenated) // usable) * usable
    input_ids: list[object] = []
    masks: list[object] = []
    for i in range(0, total, usable):
        input_ids.append([bos] + concatenated[i : i + usable] + [eos])
        masks.append(torch.ones(block_size))
    return {"input_ids": input_ids, "attention_mask": masks}


def _load_hf_dataset(dataset_name: str, cache_dir: str) -> datasets.DatasetDict:
    """Load a HuggingFace dataset, resolving known aliases."""
    hf_id = text_data_name_dict.get(dataset_name, dataset_name)
    if hf_id is None:
        raise ValueError(f"Dataset '{dataset_name}' has no HF identifier; handle separately.")
    return datasets.load_dataset(hf_id, cache_dir=cache_dir, trust_remote_code=True)


def get_text_dataloaders(
    config: DictConfig,
    tokenizer: transformers.PreTrainedTokenizer,
    skip_train: bool = False,
    skip_valid: bool = False,
    valid_seed: int | None = None,
) -> tuple[torch.utils.data.DataLoader | None, torch.utils.data.DataLoader | None]:
    """Build train/validation dataloaders for text datasets.

    Each returned loader carries a ``.tokenizer`` attribute for downstream use.
    """
    data_name: str = config.data.data
    cache_dir: str = config.data.cache_dir
    wrap: bool = config.data.wrap
    block_size: int = getattr(config.model, "length", 1024)
    batch_size: int = config.train.batch_size
    num_proc: int = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 4

    ds = _load_hf_dataset(data_name, cache_dir)

    train_split = "train"
    valid_split = "test" if data_name in ("lm1b", "text8") else "validation"
    valid_name: str = getattr(config.data, "valid", data_name)
    valid_ds = _load_hf_dataset(valid_name, cache_dir) if valid_name != data_name else ds

    eos_id: int = tokenizer.encode(tokenizer.eos_token)[0]
    bos_id: int = tokenizer.encode(tokenizer.bos_token)[0]

    # Auto-detect text column
    cols = ds[train_split].column_names
    text_col = "text" if "text" in cols else ("sentence" if "sentence" in cols else cols[0])

    def tokenize_fn(examples: dict[str, list[str]]) -> dict[str, object]:
        text = examples[text_col]
        if wrap:
            tokens = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            return {"input_ids": [t + [eos_id] for t in tokens["input_ids"]]}
        result: dict[str, object] = tokenizer(
            text,
            max_length=block_size,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        return result

    group_fn = functools.partial(_group_texts, block_size=block_size, bos=bos_id, eos=eos_id)

    def _prepare_split(split_ds: datasets.Dataset) -> datasets.Dataset:
        tok_ds = split_ds.map(
            tokenize_fn,
            batched=True,
            num_proc=num_proc,
            remove_columns=split_ds.column_names,
            desc="Tokenizing",
        )
        if wrap:
            tok_ds = tok_ds.map(group_fn, batched=True, num_proc=num_proc, desc="Grouping")
        return tok_ds.with_format("torch")

    train_set = None if skip_train else _prepare_split(ds[train_split])
    valid_set = None if skip_valid else _prepare_split(valid_ds[valid_split])

    num_workers = 16 // max(1, torch.cuda.device_count())

    if train_set is not None:
        train_loader: torch.utils.data.DataLoader | None = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
        train_loader.tokenizer = tokenizer  # type: ignore[union-attr]
    else:
        train_loader = None

    if valid_set is not None:
        shuffle_valid = valid_seed is not None
        generator = torch.Generator().manual_seed(valid_seed) if valid_seed is not None else None
        valid_loader: torch.utils.data.DataLoader | None = torch.utils.data.DataLoader(
            valid_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle_valid,
            generator=generator,
            pin_memory=True,
        )
        valid_loader.tokenizer = tokenizer  # type: ignore[union-attr]
    else:
        valid_loader = None

    return train_loader, valid_loader
