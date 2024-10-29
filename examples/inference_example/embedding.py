import argparse
from pathlib import Path

from scaling.core.logging import logger
from scaling.transformer.inference import TransformerInferenceModule


def main(checkpoint_dir: str, vocab_file: str | Path | None) -> None:
    if vocab_file is not None:
        vocab_file = Path(vocab_file)
    model = TransformerInferenceModule.from_checkpoint(
        checkpoint_dir=Path(checkpoint_dir),
        devices=[0],
        vocab_file=vocab_file,
    )
    try:
        while True:
            text = input("Please enter the text to encode (Press Ctrl+C to exit):\n")
            output = model.encode([text])
            logger.info(output)
    except KeyboardInterrupt:
        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--vocab_file", default=None, type=str)
    args = parser.parse_args()
    main(args.checkpoint_dir, args.vocab_file)
