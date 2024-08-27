import json
from pathlib import Path

from tokenizers import Tokenizer as HF_Tokenizer  # type: ignore


class Tokenizer:
    """
    Wrapper around HF tokenizer to be able to exchange easily at a later point
    """

    def __init__(self, tokenizer: HF_Tokenizer) -> None:
        self.tokenizer = tokenizer

        self.eos_token_id: int
        if self.tokenizer.token_to_id("<|endoftext|>") is None:
            self.eos_token_id = self.tokenizer.token_to_id("</s>")
        else:
            self.eos_token_id = self.tokenizer.token_to_id("<|endoftext|>")
        assert self.eos_token_id is not None

    @classmethod
    def from_file(cls, filename: str) -> "Tokenizer":
        tokenizer = HF_Tokenizer.from_file(str(filename))
        return cls(tokenizer=tokenizer)

    @classmethod
    def from_str(cls, json: str) -> "Tokenizer":
        tokenizer = HF_Tokenizer.from_str(json)
        return cls(tokenizer=tokenizer)

    @classmethod
    def default(cls) -> "Tokenizer":
        """
        Initialization as default Aleph Alpha tokenizer used for most model trainings.
        """
        filename = Path(__file__).parent / "alpha-001-128k.json"
        tokenizer = HF_Tokenizer.from_file(str(filename))
        return cls(tokenizer=tokenizer)

    def __len__(self) -> int:
        """
        Returns the vocab size of the tokenizer
        """
        return self.tokenizer.get_vocab_size(with_added_tokens=False)

    @property
    def vocab_size(self) -> int:
        return len(self)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """
        converts a string into token ids
        """
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        converts a list of token ids to a string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def load_tokenizers(tokenizer_file: str | Path) -> tuple[Tokenizer, Tokenizer]:
    # load tokenizer
    tokenizer_file = Path(tokenizer_file)
    tokenizer = Tokenizer.from_file(str(tokenizer_file))

    # for tasks that require tokenization of text within the prompt or completion (e.g. stop-tokens)
    # a tokenizer is used that does not add a whitespace at the beginning
    tokenizer_definition = json.load(open(tokenizer_file, "r", encoding="UTF-8"))

    if "llama" not in str(tokenizer_file):
        if tokenizer_definition["pre_tokenizer"] and tokenizer_definition["pre_tokenizer"]["add_prefix_space"]:
            tokenizer_definition["pre_tokenizer"]["add_prefix_space"] = False
            tokenizer_as_json_str = json.dumps(tokenizer_definition)

            tokenizer_no_prefix_space = Tokenizer.from_str(tokenizer_as_json_str)
        else:
            tokenizer_no_prefix_space = tokenizer

    else:
        if tokenizer_definition["pre_tokenizer"] and tokenizer_definition["pre_tokenizer"]["add_prefix_space"]:
            tokenizer_definition["pre_tokenizer"]["add_prefix_space"] = False

        if tokenizer_definition["decoder"] and tokenizer_definition["decoder"]["type"] == "Sequence":
            # hacky way of treating llama 2
            tokenizer_definition["decoder"]["decoders"] = [
                d
                for d in tokenizer_definition["decoder"]["decoders"]
                if not (d.get("content", "") == " " and d.get("type", "") == "Strip")
            ]

        if tokenizer_definition["normalizer"] and tokenizer_definition["normalizer"]["type"] == "Sequence":
            # hacky way of treating llama 2
            tokenizer_definition["normalizer"]["normalizers"] = [
                n for n in tokenizer_definition["normalizer"]["normalizers"] if not (n.get("type", "") == "Prepend")
            ]

        tokenizer_as_json_str = json.dumps(tokenizer_definition)
        tokenizer_no_prefix_space = Tokenizer.from_str(tokenizer_as_json_str)

    return tokenizer, tokenizer_no_prefix_space
