import torch
import yaml
import argparse


def read_yaml_config(file_path):
    """Reads a YAML configuration file and returns a dictionary of the settings."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def convert_to_args(config):
    """Converts a nested dictionary to a list of command-line arguments."""

    args = {}
    for section, settings in config.items():
        for key, value in settings.items():
            args[key] = value
    return args


def set_conf(config_file):
    yaml_file_path = config_file
    config = read_yaml_config(yaml_file_path)
    args = convert_to_args(config)
    return args


def parse_args():
    """ Parsing the Arguments for the Advertisement Generation Project"""
    parser = argparse.ArgumentParser(description="Configuration arguments for advertisement generation:")
    parser.add_argument('--config_type',
                        type=str,
                        required=True,
                        help='Choose among ARGS for commandline arguments, DEFAULT for default values, or YAML for '
                             'config file')
    parser.add_argument('--config_path',
                        type=str,
                        default=None,
                        help='The path to the config file if config_type is YAML')
    parser.add_argument('--evaluation_type',
                        type=str,
                        default='topic_based')
    parser.add_argument('--result_path',
                        type=str,
                        default='../experiments/results',
                        help='The path to the folder for saving the results')
    parser.add_argument('--result_file',
                        type=str,
                        required=True,
                        help='the file path relative to the result_path')
    parser.add_argument('--text_input_type',
                        type=str,
                        default='LLM')
    parser.add_argument('--with_sentiment',
                        type=bool,
                        default=False)
    parser.add_argument('--with_topic',
                        type=bool,
                        default=False)
    parser.add_argument('--llm_prompt',
                        type=str,
                        default='LLM_input.jinja',
                        help='LLM input prompt template file name.')
    parser.add_argument('--T2I_prompt',
                        type=str,
                        default='LLM.jinja',
                        help='T2I input prompt template file name.')
    parser.add_argument('--data_path',
                        type=str,
                        default='../Data/PittAd',
                        help='Path to the root of the data'
                        )
    parser.add_argument('--product_images',
                        type=str,
                        default='product_images',
                        help='path to the typical images for the advertised product in each advertisement'
                        )
    parser.add_argument('--train_set_QA',
                        type=str,
                        default='train/Action_Reason_statements.json',
                        help='If the model is fine-tuned, relative path to the train-set QA from root path')
    parser.add_argument('--train_set_images',
                        type=str,
                        default='train_images',
                        help='If the model is fine-tuned, relative path to the train-set Images from root path')
    parser.add_argument('--test_set_QA',
                        type=str,
                        default='train/Action_Reason_statements.json',
                        help='Relative path to the QA file for action-reasons from root path'
                        )
    parser.add_argument('--test_set_images',
                        type=str,
                        default='train_images',
                        help='Relative path to the original images for the test set from root')
    parser.add_argument('--description_file',
                        type=str,
                        default=None,
                        help='Path to the description file for the T2I input.')
    return parser.parse_args()


def get_args():
    args = parse_args()
    if args.config_type == 'YAML':
        args = set_conf(args.config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    print("Arguments are:\n", args, '\n', '-'*40)
    return args




