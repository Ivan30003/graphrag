import argparse
from pathlib import Path
import os
from shutil import copy

from files_utils import read_yaml_file
from pipeline import Pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=Path, required=True)

    args = parser.parse_args()
    config_path = args.config_path
    
    print("START")
    config = read_yaml_file(config_path)
    os.makedirs(config['working_dir'], exist_ok=True)
    copy(config_path, config['working_dir'])
    pipeline = Pipeline(config)
    pipeline.launch()

    print("END")
