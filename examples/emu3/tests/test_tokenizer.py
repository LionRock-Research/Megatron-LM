import pytest

from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer
from transformers import AutoTokenizer

EMU3_MODEL_DIR = "/data/models/Emu3-Gen"

@pytest.mark.parametrize("sentence", ["a portrait of young girl.", "什么东西", "life is a fucking movie"])
def test_tokenizer(sentence):
    tokenizer_megatron = _HuggingFaceTokenizer(EMU3_MODEL_DIR)
    tokenizer_huggingface = AutoTokenizer.from_pretrained(EMU3_MODEL_DIR, trust_remote_code=True)
    assert tokenizer_megatron.tokenize(sentence) == tokenizer_huggingface.encode(sentence)
