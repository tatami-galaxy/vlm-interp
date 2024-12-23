import os
import subprocess


def main():

    req = [
        'matplotlib', "bitsandbytes",
        'transformers', 'datasets', 'accelerate',
        'evaluate', 'tensorboard', 'wandb',
    ]


    # directorries
    os.mkdir('data')
    os.mkdir('data/raw')
    os.mkdir('data/processed')

    os.mkdir('models')

    # install packages
    for package in req:
        subprocess.run(["pip", "install", package])
    subprocess.run(["pip", "install", "flash-attn", "--no-build-isolation"])


if __name__ == "__main__":
    main()
