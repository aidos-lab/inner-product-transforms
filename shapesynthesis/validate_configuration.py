"""
Simple script to validate the configuration. This ensures no keys are missing,
catches typos in the config files and raises errors when references are missing.
"""

import argparse

from loaders import load_config, validate_configuration

parser = argparse.ArgumentParser()
parser.add_argument("INPUT", type=str, help="Input configuration")

args = parser.parse_args()

run_config, run_config_dict = load_config(args.INPUT)

# Validate the data.
validate_configuration(run_config_dict)
