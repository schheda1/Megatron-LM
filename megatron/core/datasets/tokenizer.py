# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

from abc import ABC
from abc import abstractmethod

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer



def build_tokenizer(tokenizer_type='NullTokenizer', vocab_size=100):
    """Initialize tokenizer."""
    #if args.rank == 0:
    #    print('> building {} tokenizer ...'.format(args.tokenizer_type),
    #          flush=True)

    if tokenizer_type == 'NullTokenizer':
        assert vocab_size is not None
        tokenizer = _NullTokenizer(vocab_size)
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(tokenizer_type))

    # Add vocab size (if not already set from a checkpoint).
    #if getattr(args, "padded_vocab_size", None) is None:
    #    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size,
    #                                                      args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * \
        args.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after



class _NullTokenizer(MegatronTokenizer):
    def __init__(self, vocab_size):
        super().__init__(None, vocab_size=vocab_size)
        self._vocab_size_without_eod = int(vocab_size)
        self._eod_id = 0  #self._vocab_size_without_eod

    def tokenize(self, text):
        return [int(x) for x in text.split(' ')]

    def detokenize(self, ids):
        text = [str(x) for x in ids]
        return ' '.join(text)

    @property
    def vocab_size(self):
        return self._vocab_size_without_eod + 1

    @property
    def vocab(self):
        raise NotImplementedError

    @property
    def inv_vocab(self):
        raise NotImplementedError

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eod_id

    @property
    def additional_special_tokens_ids(self):
        return None
