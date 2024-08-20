from types import MethodType

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizer

from ..extras import get_logger


logger = get_logger(__name__)


def patch_tokenizer(tokenizer: PreTrainedTokenizer) -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)
