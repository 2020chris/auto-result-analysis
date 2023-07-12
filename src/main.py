import os
import yaml
import logging
import torch 
import transformers
import argparse
from typing import Text
import json
from load_data import load_data
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from data_collator import createMultimodalDataCollator
from model import initialize
from train import train

def main(config_path:Text) -> None:
    transformers.logging.set_verbosity_error()
    logging.basicConfig(level=logging.INFO)

    with open(config_path) as conf_f:
        config = yaml.safe_load(conf_f)
    
    if config["base"]["use_cuda"]:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' # SET ONLY 1 GPU DEVICE
    else:
        device =  torch.device('cpu')


    df = pd.read_csv(os.path.join(config["data"]["data_dir"], config["data"]["full_dataset"]))
    df_reduc = df[['StepDescription', 'image_id', 'screenshot', 'IssueType']]
    train_df, test_df = train_test_split(df_reduc, test_size=0.2, random_state=42)
    train_df.to_csv(os.path.join(config["data"]["data_dir"], config["data"]["train_dataset"]))
    test_df.to_csv(os.path.join(config["data"]["data_dir"], config["data"]["test_dataset"]))

    dataset = load_dataset("csv", data_files = {
            "train": os.path.join(config["data"]["data_dir"], config["data"]["train_dataset"]),
            "test": os.path.join(config["data"]["data_dir"], config["data"]["test_dataset"])
        }
    )

    labels = df['IssueType']

    logging.info("Loaded preprocessed Blucumber test data")
    
    multimodal_collator = createMultimodalDataCollator(config)
    logging.info("Initialized data collator")

    model = initialize(config, labels)
    logging.info("Initialized multi-modal model for classification")


    logging.info("Training started....")
    training_metrics, eval_metrics = train(config, device, dataset, multimodal_collator, model)

    logging.info("Training completed.")

    os.makedirs(config["metrics"]["metrics_folder"], exist_ok=True)
    metrics = {**training_metrics[2], **eval_metrics}

    metrics_path = os.path.join(config["metrics"]["metrics_folder"], config["metrics"]["metrics_file"])
    json.dump(
        obj=metrics,
        fp=open(metrics_path, 'w'),
        indent=4
    )

    logging.info("Metrics saved")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)   #need to pass in yaml file 
    args = arg_parser.parse_args()
    main(args.config)