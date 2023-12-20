import os
import sys

import yaml

project_dir = os.environ.get('PROJECT_DIR')

# Load the YAML configuration file
with open('config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
