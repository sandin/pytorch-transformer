from config import get_config
from train import train_model

def main():
    print("Hello from pytorch-transformer!")
    cfg = get_config()
    cfg['model_folder'] = './weights'
    cfg['tokenizer_file'] = './vocab/tokenizer_{0}.json'
    cfg['batch_size'] = 24
    cfg['num_epochs'] = 100
    cfg['preload'] = None

    train_model(cfg)


if __name__ == "__main__":
    main()
