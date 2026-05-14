import json
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("configs/wmt19.json")

def get_config(config_path=DEFAULT_CONFIG_PATH):
    config = {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_type": "WordLevel",  # or "BPE"
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with config_file.open("r", encoding="utf-8") as f:
                file_config = json.load(f)
            if not isinstance(file_config, dict):
                raise ValueError(f"Config file must contain a JSON object: {config_file}")
            config.update(file_config)
        else:
            raise FileNotFoundError(f"Config file not found: {config_file}")

    return config

def get_weights_file_path(config, epoch: str):
    model_folder = Path(config['model_folder']) / config['datasource']
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = Path(config['model_folder']) / config['datasource']
    model_filename = f"{config['model_basename']}*.pt"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
