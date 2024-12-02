import os
import subprocess


def main():

    req = ['transformers', 'datasets', 'accelerate', 'qwen_vl_utils']
    # install packages
    for package in req:
        subprocess.run(["pip", "install", package])
    subprocess.run(["pip", "install", "flash-attn", "--no-build-isolation"])

    # rclone -> test
    subprocess.run(["curl", "https://rclone.org/install.sh"])
    subprocess.run(["bash"])
    subprocess.run(["rclone", "config"])


if __name__ == "__main__":
    main()
