"""
This module builds up tokenization template for RNAErnie families.
Author: wangning(nwang227-c@my.cityu.edu.hk)
Date  : 2024/05/29 11:30
"""

from __future__ import annotations

import os
import collections
from typing import List, Tuple, Optional

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class RNATokenizer(PreTrainedTokenizer):
    """
    Constructs a base RNA tokenizer.
    """

    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.txt"}

    def __init__(
        self,
        vocab_file,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        bos_token: str = "[CLS]",
        eos_token: str = "[EOS]",
        mask_token: str = "[MASK]",
        do_upper_case: bool = True,
        replace_T_with_U: bool = True,
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.do_upper_case = do_upper_case
        self.replace_T_with_U = replace_T_with_U

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text: str) -> list[str]:
        if self.do_upper_case:
            text = text.upper()
        if self.replace_T_with_U:
            text = text.replace("T", "U")
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def token_to_id(self, token: str) -> int:
        # type: ignore[arg-type]
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: List[int] | None = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            if self.eos_token_id is None:
                return cls + token_ids_0
            return cls + token_ids_0 + eos
        if self.eos_token_id is None:
            raise ValueError(
                "Cannot tokenize multiple sequences when EOS token is not set!")
        return cls + token_ids_0 + sep + token_ids_1 + eos

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: List[int] | None = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        mask = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]
        return mask

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") +
                self.vocab_files_names["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix +
                          "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
