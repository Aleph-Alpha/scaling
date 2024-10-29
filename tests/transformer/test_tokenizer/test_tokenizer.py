from pathlib import Path

import pytest

from scaling.transformer.tokenizer import Tokenizer, load_tokenizers

TEST_DEFAULT_TOKENIZER_FILE = Path(__file__).parents[1].absolute() / "files" / "alpha-001-128k.json"
TEST_LLAMA2_TOKENIZER_FILE = Path(__file__).parents[1].absolute() / "files" / "llama2-tokenizer.json"
TEST_LLAMA3_TOKENIZER_FILE = Path(__file__).parents[1].absolute() / "files" / "llama-3.1-8B-tokenizer.json"


# pytest tests/test_tokenizer/test_tokenizer.py::test_tokenization -s
@pytest.mark.cpu
@pytest.mark.parametrize(
    "text",
    [
        "This is a test.",
        "Another example.",
        "Also strange stuff: daölfijapwe9o8rihjpöalsdifjapsdoifj",
        "And maybe an emojy? 😀",
    ],
)
def test_tokenization(text):
    # load tokenizer

    tokenizer = Tokenizer.from_file(str(TEST_DEFAULT_TOKENIZER_FILE))

    # get vocab size
    vocab_size = len(tokenizer)
    assert isinstance(vocab_size, int), f"vocab size is not an int, got {vocab_size}"

    # encode
    token_ids = tokenizer.encode(text)

    # decode
    decoded = tokenizer.decode(token_ids)

    assert text.strip() == decoded.strip(), "decoded text is not equal to the input text"


# pytest tests/test_tokenizer/test_tokenizer.py::test_eos_token_id -s
def test_eos_token_id():
    tokenizer = Tokenizer.from_file(str(TEST_DEFAULT_TOKENIZER_FILE))

    assert tokenizer.eos_token_id == 0, tokenizer.eos_token_id

    # Make sure LRU cache doesnt crash
    assert tokenizer.eos_token_id == 0, tokenizer.eos_token_id
    assert tokenizer.eos_token_id == 0, tokenizer.eos_token_id

    tokenizer = Tokenizer.from_file(str(TEST_LLAMA2_TOKENIZER_FILE))

    assert tokenizer.eos_token_id == 2, tokenizer.eos_token_id  # "</s>"

    tokenizer = Tokenizer.from_file(str(TEST_LLAMA3_TOKENIZER_FILE))

    assert tokenizer.eos_token_id == 128001, tokenizer.eos_token_id  # "<|end_of_text|>"


def test_load_default_tokenizer():
    _ = Tokenizer.default()


def test_tokenizer_no_prefix_space():
    tokenizer, tokenizer_no_prefix_space = load_tokenizers(TEST_LLAMA2_TOKENIZER_FILE)
    assert tokenizer.encode("Hello") != tokenizer_no_prefix_space.encode("Hello")
