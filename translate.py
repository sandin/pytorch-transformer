import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from config import DEFAULT_CONFIG_PATH, get_config, latest_weights_file_path, load_translation_dataset


@dataclass
class TranslationResult:
    text: str
    generated_tokens: int
    elapsed_seconds: float

    @property
    def tokens_per_second(self):
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.generated_tokens / self.elapsed_seconds


class Translator:
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        import torch
        from tokenizers import Tokenizer

        from model import build_transformer

        self.torch = torch
        self.config_path = Path(config_path)
        self.config = get_config(self.config_path)

        self.device = self._get_device()
        self.tokenizer_src = Tokenizer.from_file(str(Path(self.config['tokenizer_file'].format(self.config['lang_src']))))
        self.tokenizer_tgt = Tokenizer.from_file(str(Path(self.config['tokenizer_file'].format(self.config['lang_tgt']))))
        self.model = build_transformer(
            self.tokenizer_src.get_vocab_size(),
            self.tokenizer_tgt.get_vocab_size(),
            self.config["seq_len"],
            self.config["seq_len"],
            d_model=self.config["d_model"],
        ).to(self.device)

        model_filename = latest_weights_file_path(self.config)
        if model_filename is None:
            raise FileNotFoundError("No model weights found for this config.")

        print(f"Loading Model: {model_filename}")
        state = torch.load(model_filename, map_location="cpu")
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

    def _get_device(self):
        torch = self.torch
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        device = torch.device(device_name)

        if device_name == "cuda":
            print(f"Device name: {torch.cuda.get_device_name(device)}")
            print(f"Device memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3} GB")
        elif device_name == "mps":
            print("Device name: mps")
        else:
            print("NOTE: If you have a GPU, consider using it for training.")
            print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
            print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")

        return device

    def _synchronize_device(self):
        torch = self.torch
        device_type = self.device.type

        if device_type == "cuda":
            torch.cuda.synchronize(self.device)
        elif device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def translate(self, sentence: str, return_stats: bool = False):
        from dataset import BilingualDataset

        label = ""
        dataset_id = None
        if isinstance(sentence, int) or str(sentence).isdigit():
            dataset_id = int(sentence)
            ds = load_translation_dataset(self.config, split="all")
            ds = BilingualDataset(
                ds,
                self.tokenizer_src,
                self.tokenizer_tgt,
                self.config["lang_src"],
                self.config["lang_tgt"],
                self.config["seq_len"],
            )
            sentence = ds[dataset_id]["src_text"]
            label = ds[dataset_id]["tgt_text"]

        seq_len = self.config["seq_len"]

        with self.torch.no_grad():
            source = self.tokenizer_src.encode(sentence)
            source = self.torch.cat([
                self.torch.tensor([self.tokenizer_src.token_to_id("[SOS]")], dtype=self.torch.int64),
                self.torch.tensor(source.ids, dtype=self.torch.int64),
                self.torch.tensor([self.tokenizer_src.token_to_id("[EOS]")], dtype=self.torch.int64),
                self.torch.tensor([self.tokenizer_src.token_to_id("[PAD]")] * (seq_len - len(source.ids) - 2), dtype=self.torch.int64),
            ], dim=0).to(self.device)
            source_mask = (source != self.tokenizer_src.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).int().to(self.device)
            encoder_output = self.model.encode(source, source_mask)

            decoder_input = self.torch.empty(1, 1).fill_(self.tokenizer_tgt.token_to_id("[SOS]")).type_as(source).to(self.device)

            if label != "":
                print(f"{f'ID: ':>12}{dataset_id}")
            print(f"{f'SOURCE: ':>12}{sentence}")
            if label != "":
                print(f"{f'TARGET: ':>12}{label}")
            print(f"{f'PREDICTED: ':>12}", end="")

            generated_tokens = 0
            self._synchronize_device()
            start_time = time.perf_counter()

            while decoder_input.size(1) < seq_len:
                decoder_mask = self.torch.triu(self.torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(self.torch.int).type_as(source_mask).to(self.device)
                out = self.model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

                prob = self.model.project(out[:, -1])
                _, next_word = self.torch.max(prob, dim=1)
                next_word_id = next_word.item()
                decoder_input = self.torch.cat([decoder_input, self.torch.empty(1, 1).type_as(source).fill_(next_word_id).to(self.device)], dim=1)
                generated_tokens += 1

                print(f"{self.tokenizer_tgt.decode([next_word_id])}", end=" ")

                if next_word_id == self.tokenizer_tgt.token_to_id("[EOS]"):
                    break

            self._synchronize_device()
            elapsed_seconds = time.perf_counter() - start_time

        predicted_text = self.tokenizer_tgt.decode(decoder_input[0].tolist())
        print()
        result = TranslationResult(predicted_text, generated_tokens, elapsed_seconds)
        print(f"{f'TOK/S: ':>12}{result.tokens_per_second:.2f}")

        if self.config.get("score_bleu", False):
            if label != "":
                from sacrebleu import sentence_bleu

                bleu = sentence_bleu(predicted_text, [label])
                print(f"{f'BLEU: ':>12}{bleu.score:.2f}")
            else:
                print("BLEU scoring is enabled, but no reference translation is available.")

        if return_stats:
            return result
        return predicted_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", nargs="?", default="I am not a very good a student.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file")
    args = parser.parse_args()

    translator = Translator(args.config)
    translator.translate(args.sentence)

if __name__ == "__main__":
    main()
