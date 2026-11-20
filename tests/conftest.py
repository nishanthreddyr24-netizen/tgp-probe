"""A tiny, randomly initialised Llama + word-level tokenizer, built offline."""

import pytest
import torch

WORDS = (
    "the a of to in is was and on for by with from at as it that this "
    "amazon bezos musk founded company paris france capital city river "
    "music video japan dj equipment summary article question answer passage "
    "red blue green one two three four five six seven eight nine ten"
).split()


@pytest.fixture(scope="session")
def tiny_model_and_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    specials = ["<pad>", "<s>", "</s>", "<unk>"]
    vocab = {w: i for i, w in enumerate(specials + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", vocab["<s>"])]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="</s>",
        unk_token="<unk>", pad_token="<pad>",
    )

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256, pad_token_id=0,
    )
    model = LlamaForCausalLM(config).eval()
    return model, tokenizer


@pytest.fixture()
def probe(tiny_model_and_tokenizer):
    from tgp_probe import TGPProbe

    model, tokenizer = tiny_model_and_tokenizer
    return TGPProbe(model, tokenizer, model_name="tiny-llama")
