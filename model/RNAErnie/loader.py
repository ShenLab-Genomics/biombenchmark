from typing import Any, Dict, TypedDict

from transformers import AutoConfig, AutoModelForMaskedLM

from .patcher import patch_tokenizer

from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from .tokenization_rnaernie import RNAErnieTokenizer


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"


def _get_init_kwargs(model_args) -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        # "revision": model_args.model_revision,
        # "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args) -> TokenizerModule:
    r"""
    Loads pretrained tokenizer.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)

    try:
        tokenizer = RNAErnieTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:
        tokenizer = RNAErnieTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )

    patch_tokenizer(tokenizer)
    return {"tokenizer": tokenizer}
