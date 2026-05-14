import argparse

from config import DEFAULT_CONFIG_PATH, get_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file")
    args = parser.parse_args()

    from train import train_model

    print("Hello from pytorch-transformer!")
    cfg = get_config(args.config)

    train_model(cfg)


if __name__ == "__main__":
    main()
