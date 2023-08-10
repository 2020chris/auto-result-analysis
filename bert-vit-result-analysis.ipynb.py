# Databricks notebook source
# MAGIC %md ### Automated Result Analysis with BERT+ViT fusion model - TEXT + IMAGE MODEL
# MAGIC
# MAGIC #### Intro: 
# MAGIC Over hundreds of Blucumber Regression Tests are run weekly to test various Features (composed of multiple scenarios) of different Cvent products and manually analyzed by a dedicated Prod team. Part of the analysis is classifying errors observed in the run scenario steps into one of a few different target status buckets(ie. SyncIssue, DataIssue, HardError, WebDriverIssue, ect.). This manual analysis of regression test results is evidently a time-intensive task and could be considered a major bottleneck in the Cvent SDLC. A rule-based auto-analysis feature is currently integrated into the QE Portal but has a large misclassification rate. 
# MAGIC
# MAGIC #### Goals: 
# MAGIC Develop and integrate a new QE portal system for automated regression test analysis which leverages statistical machine learning methods to create a probabilistic model for classifying test case failure modes. Instead of defining the predictive model manually, we will build this service by using a large MySQL dataset of testexecution metadata and steplogs containing the results of Bluecumber tests on historical code changes and then apply standard ML techniques for classification. Scalable deployment using the Triton Inference Server: https://developer.nvidia.com/triton-inference-server. Intended to be a generalizable approach applicable for other tests including Wdio/API testing.

# COMMAND ----------

PATH = "/dbfs/mnt/s3_mount/qe_automation/data/"
aws_bucket_name = "ml-hub-workspace-root-sg50"
mount_name = "s3_mount"
mount_point = "/mnt/%s" % mount_name
try:
    dbutils.fs.mount("s3a://%s" % aws_bucket_name, mount_point)
except Exception as e:
    print("Already mounted!")
#dbutils.fs.mount(f"s3a://{aws_bucket_name}", f"/mnt/{mount_name}")
display(dbutils.fs.ls(f"/mnt/{mount_name}"))

# COMMAND ----------

# MAGIC %pip install --extra-index-url https://pypi.nvidia.com cudf-cu11
# MAGIC %pip install cuml-cu11 --extra-index-url https://pypi.nvidia.com
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import cudf
import cuml

# COMMAND ----------

# MAGIC %  /dbfs/mnt/s3_mount/test.npy

# COMMAND ----------

mount_point

# COMMAND ----------

import os 
if not os.path.exists("/dbfs/mnt/s3_mount/weights"):   
    os.makedirs("/dbfs/mnt/s3_mount/weights")

# COMMAND ----------

ls /dbfs/mnt/s3_mount/

# COMMAND ----------

# MAGIC %pip install bitsandbytes
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())


# COMMAND ----------

# MAGIC %pip install "pynvml" "accelerator" "mlflow"
# MAGIC %pip install opencv-python
# MAGIC dbutils.library.restartPython

# COMMAND ----------

import yaml
with open('params.yaml', 'r') as f:
    config = yaml.safe_load(f)
config

# COMMAND ----------

# MAGIC %md ### Data preprocessing

# COMMAND ----------

import io
import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
# os.makedirs('images', exists_ok=True)

def deserialize(row):
    try:
        binary_data = row['screenshot']
        image_stream = io.BytesIO(binary_data)
        image = Image.open(image_stream)
        execution_id = row['ExecutionId']
        execution_item_id = row['ExecutionItemID']
        step_log_id = row['StepLogID']
        image_id = f'img_{execution_id}_{execution_item_id}_{step_log_id}.png'
       # image_path = os.path.join('dataset', 'images', image_id)
      #  image.save(image_path)
        return image_id
    except UnidentifiedImageError:
        return None
    
# Apply the function to the DataFrame and assign the results to a new column 'image_path'
df = pd.read_parquet('/dbfs/FileStore/tables/data_optimized.parquet')
df['image_id'] = df.apply(deserialize, axis=1)
print("Original # rows: ", len(df))

# COMMAND ----------

df.dropna(subset=['image_id', 'screenshot', 'IssueType'], inplace=True)
print("Processed # rows: ", len(df))
# Add a new column 'image_id' based on the image_path
#df_subset['image_id'] = df_subset['image_path'].apply(lambda path: os.path.basename(path))
df.to_parquet('/dbfs/FileStore/tables/data_optimized_proc.parquet', compression='brotli')
# Print the updated DataFrame
#df_subset.to_csv('subset.csv')
df.head()

# COMMAND ----------

# normalize
import pandas as pd

encode_dict = {}
def encode_label(x):
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]

#df = pd.read_csv('/dbfs/FileStore/shared_uploads/Chris.Jose@cvent.com/subset.csv')
df = pd.read_parquet('/dbfs/FileStore/tables/data_optimized_proc.parquet')
df["CAT"] = df["IssueType"].map(lambda x: encode_label(x))
df

# COMMAND ----------

df.StepDescription[0]

# COMMAND ----------

import torchvision.transforms as transforms
from PIL import Image
import io
transform = transforms.Compose([transforms.Resize(124)])
transformed_img = transform(Image.open(io.BytesIO(df["screenshot"][0])))
transformed_img

# COMMAND ----------

Image.open(io.BytesIO(df["screenshot"][0]))

# COMMAND ----------

import matplotlib.pyplot as plt
counts = df.IssueType.value_counts()
fig = plt.figure(figsize=(23,10))
plt.bar(df.IssueType.unique(), counts, width=0.3)
plt.xlabel("Ground truth labels")
plt.ylabel("Counts")
plt.title("Label Frequencies")
plt.show()

# COMMAND ----------

import cv2
from PIL import Image
import io
import numpy as np
count = 0
for idx, row in enumerate(df["screenshot"]):
    img = Image.open(io.BytesIO(row))
    # h,w = img.shape[:2]
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
   # print('height: {} width: {}'.format(h, w))
    if h != 940 or w != 1920:
        #print('not consistent')
        print('idx: {} height: {} width: {}'.format(idx, h, w))
   # count+=1
    #if(count == 3):
     #   break

df.CAT.value_counts()

# COMMAND ----------

from transformers import AutoTokenizer

qs = df["StepDescription"].tolist()
tokenizer_cvent =  AutoTokenizer.from_pretrained(
            "vives/distilbert-base-uncased-finetuned-cvent-2019_2022")
max_len = max([len(tokenizer_cvent(q)["input_ids"]) for q in qs])
print(max_len)

# COMMAND ----------

import pandas as pd
from transformers import BertTokenizer

exceeds_tok_limit = []
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


batch_size = 8  # You can adjust this based on your GPU memory capacity
num_batches = (len(df) - 1) // batch_size + 1

for i in range(num_batches):
    batch_start = i * batch_size
    batch_end = min((i + 1) * batch_size, len(df))

    batch_texts = df.iloc[batch_start:batch_end]["StepDescription"].tolist()
    batch_tokens = tokenizer.batch_encode_plus(batch_texts, add_special_tokens=True, truncation=True, padding='max_length', max_length=512, return_overflowing_tokens=True)
    count = 0
    for j, tokens in enumerate(batch_tokens["overflowing_tokens"]):
        if tokens:
            #exceeds_tok_limit.append(df.iloc[batch_start + j])
            print(tokens)

print(exceeds_tok_limit)

# COMMAND ----------

from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle) 
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
print_gpu_utilization()

# COMMAND ----------

# MAGIC %md ### Data tokenization

# COMMAND ----------

# Convert raw images and questions into inputs for featurization, to be fed batchwise into classifier

import os
import io
from dataclasses import dataclass
from typing import Dict, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler, Subset
from transformers import AutoTokenizer, AutoFeatureExtractor, DistilBertTokenizer
import yaml
import logging
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

@dataclass
class BluecumberDataModule:
    def __init__(self, config: Dict, df):
        self.config = config
        self.data = df
        self.len = len(df)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       # self.transform = transforms.Compose([transforms.Resize(124)])
                                            # transforms.ToTensor(),
                                             #transforms.Normalize(mean=[0.8334760665893555, 0.8291550278663635, 0.8283498883247375], std=[0.12601536512374878, 0.12896546721458435, 0.13210657238960266])])
        if config["model"]["text_encoder"] == "distilbert-base-uncased":
            self.tokenizer = DistilBertTokenizer.from_pretrained(config["model"]["text_encoder"])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
       # self.model = DistilBertModel.from_pretrained(config["model"]["text_encoder"], output_hidden_states=True).to(self.device)
        self.preprocessor = AutoFeatureExtractor.from_pretrained(config["model"]["image_encoder"])
        img_mean = self.preprocessor.image_mean
        img_std = self.preprocessor.image_std
        self.transform = transforms.Compose([transforms.Resize(124),
                                             transforms.ToTensor()])
                                             #transforms.Normalize(mean=img_mean, std=img_std)])

    def tokenize_text(self, texts: str):
        encoded_text = self.tokenizer.encode_plus(
            texts,
            padding=self.config["tokenizer"]["padding"],
            max_length=self.config["tokenizer"]["max_length"],
            truncation=self.config["tokenizer"]["truncation"],
            return_tensors='pt',
            return_token_type_ids=self.config["tokenizer"]["return_token_type_ids"],
            return_attention_mask=self.config["tokenizer"]["return_attention_mask"],
        )
        return {
            "input_ids": torch.tensor(encoded_text['input_ids'].squeeze(), dtype=torch.long),
            "token_type_ids": torch.tensor(encoded_text['token_type_ids'].squeeze(), dtype=torch.long),
            "attention_mask": torch.tensor(encoded_text['attention_mask'].squeeze(), dtype=torch.long)
        }

    def tokenize_and_pool(self, texts: str):
        toks = self.tokenizer.encode_plus(
            texts,
            add_special_tokens=False,
            padding=self.config["tokenizer"]["padding"],
            max_length=self.config["tokenizer"]["max_length"],
            truncation=self.config["tokenizer"]["truncation"],
            return_tensors='pt',
          #  return_token_type_ids=self.config["tokenizer"]["return_token_type_ids"],
            return_attention_mask=self.config["tokenizer"]["return_attention_mask"],
            
        )

        return {
            "input_ids":toks['input_ids'].squeeze(),
            "attention_mask": toks['attention_mask'].squeeze()
        }

    def preprocess_image(self, binary_data):
        
        transformed_img = self.transform(Image.open(io.BytesIO(binary_data)).convert('RGB'))
        processed_images = self.preprocessor(
            images=[ transformed_img],
            return_tensors="pt",
        )
         # print("---IMG MEAN: ", transformed_img.mean(dim=(1,2)))
       # print("---IMG STD: ", transformed_img.std(dim=(1,2)))
      #  mean = processed_images['pixel_values'].mean(dim=0)
      #  std = processed_images['pixel_values'].std(dim=0)
      #  normalize_transform = transforms.Normalize(mean=mean, std=std)
      #  normalized_features = normalize_transform(processed_images['pixel_values'])
      #  print(normalized_features.squeeze().float())

        return {
            "pixel_values": processed_images['pixel_values'].squeeze().float(),
        }
        
    """         
    def __call__(self, raw_batch_dict):
        return {
            **self.tokenize_text(
                raw_batch_dict['StepDescription']
                if isinstance(raw_batch_dict, dict) else
                [i['StepDescription'] for i in raw_batch_dict]
            ),
            **self.preprocess_images(
                raw_batch_dict["image_id"]
                if isinstance(raw_batch_dict, dict) else
                [i['image_id'] for i in raw_batch_dict]
            ),
            'labels': torch.tensor(
                raw_batch_dict['IssueType']
                if isinstance(raw_batch_dict, dict) else
                [i['IssueType'] for i in raw_batch_dict],
                dtype=torch.int64
            ),
        }
    """
    def __getitem__(self, idx):
        text_results = self.tokenize_and_pool(self.data.StepDescription[idx].lower())
     #   text_results = self.tokenize_and_pool(self.data.StepDescription[idx].lower())
  #      img_results = self.preprocess_image(self.data.image_id[idx])
        img_results = self.preprocess_image(self.data.screenshot[idx])
        return {
            **text_results,
            **img_results,
            'labels': torch.tensor(self.data.CAT[idx], dtype=torch.int64)
        }
    
    def __len__(self):
        return self.len
    
def collate_fn(batch):
    # pad the batch of ragged tensors to the same len
    lengths = [len(item['input_ids']) for item in batch]
    padded_inp_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    padded_attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)

    # pack the padded tensor into a packed sequence representation
    return {
        'input_ids': padded_inp_ids, #pack_padded_sequence(padded_inp_ids, lengths, batch_first=True, enforce_sorted=False),
        'attention_mask': padded_attention_mask, #pack_padded_sequence(padded_attention_mask, lengths, batch_first=True, enforce_sorted=False),
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'lengths': lengths
    }


def createMultimodalDataCollator(config: Dict, df):
    train_df, test_df = train_test_split(df, train_size=config["data"]["train_size"], random_state=42, stratify=df.IssueType)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print("- TRAIN -\n")
    print(train_df.groupby(["CAT", "IssueType"])["CAT"].count())
    print("-----------------")
    print("- TEST -\n")
    print(test_df.groupby(["CAT", "IssueType"])["CAT"].count())

    training_set = BluecumberDataModule(config, train_df)
    testing_set = BluecumberDataModule(config, test_df)

    cat = train_df.CAT.tolist()
    class_sample_count = np.array([len(np.where(cat == t)[0]) for t in np.unique(cat)])
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in cat])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(
        samples_weight.type("torch.DoubleTensor"), len(samples_weight)
    )

    train_params = {
        "batch_size": config["train"]["per_device_train_batch_size"],
        "sampler": sampler,
        "pin_memory": True,
        "num_workers": 0,
    }

    test_params = {
        "batch_size": config["train"]["per_device_eval_batch_size"],
        "shuffle": True,
        "pin_memory": True,
        "num_workers": 0,
    }

    training_loader = DataLoader(training_set, collate_fn=collate_fn, **train_params)
    testing_loader = DataLoader(testing_set, collate_fn=collate_fn, **test_params)
    return training_loader, testing_loader, training_set, testing_set, training_set.tokenizer, training_set.preprocessor

# COMMAND ----------

# MAGIC %md ### Model Architecture

# COMMAND ----------

from typing import List, Dict, Optional
import torch
from torch import nn
from transformers import (AutoModel, DistilBertTokenizer,
    DistilBertModel,
    AutoTokenizer,
    AutoModelForMaskedLM,
)
"""
After the input sequence is tokenized and embedded as a sequence of vectors, the transformer model processes the sequence through multiple layers of self-attention and feed-forward neural networks. The final layer typically outputs a sequence of hidden states, where each hidden state corresponds to a different position in the input sequence.
"""

class MultimodalFusionModel(nn.Module):

    def __init__(self, config, num_labels: int, intermediate_dims: int, dropout: float, pretrained_text_model: str, pretrained_image_model: str):
        super(MultimodalFusionModel, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if pretrained_text_model == "distilbert-base-uncased": #"distilbert-base-cased-distilled-squad":
            self.text_encoder = DistilBertModel.from_pretrained(pretrained_text_model, output_hidden_states=True)#.to(self.device)
        elif pretrained_text_model == "vives/distilbert-base-uncased-finetuned-cvent-2019_2022":
           # self.text_encoder = AutoModel.from_pretrained(pretrained_text_model)
           self.text_encoder = AutoModelForMaskedLM.from_pretrained(pretrained_text_model, output_hidden_states=True)
        self.image_encoder = AutoModel.from_pretrained(pretrained_image_model)#.to(self.device)

        # freeze backbone layers
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.late_fusion = nn.Sequential(
            nn.Linear(in_features=self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, out_features = intermediate_dims),
            nn.ReLU(),
            nn.Dropout(dropout), 
        )
        self.pred = nn.Linear(in_features = intermediate_dims, out_features = num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def pool_embeddings(self, out, tok):
        """
        returns the pooler_output - the final hidden state of the [CLS] token - a summary representation of the input sequence

        Concatenates tokens among the output dimension via a mean reduction (e.g output of 5x728 gets reduced to 1x728)
        """
        embeddings = out["hidden_states"][-1]
        attention_mask = tok["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        return mean_pooled

    def gen_embeddings_sliding_window(self, input_ids_list, attention_mask_list):  
        """
        BERT is restricted to consuming 512 tokens per sample - extract a window from input_ids and attention_mask, append the start/separater tokens, add padding, format into dictionary containing a stacked tensors
        Other approach - truncate/select most relevant 512 tokens

        """
        max_length = self.config["tokenizer"]["max_length"]  
        list_of_segment_embeddings = []
    
        for i in range(len(input_ids_list)):  
            input_ids = input_ids_list[i]  
            attention_mask = attention_mask_list[i]  

            # split input_ids and attention_mask into chunks of size max_length-2  
            input_id_chunks = list(input_ids.split(max_length - 2))  
            mask_chunks = list(attention_mask.split(max_length - 2))

            for j in range(len(input_id_chunks)):  
                # add CLS and SEP tokens to input IDs
                input_id_chunks[j] = torch.cat([
                    torch.tensor([101]).to(self.device), input_id_chunks[j], torch.tensor([102]).to(self.device)
                ])
                # add attention tokens to attention mask
                mask_chunks[j] = torch.cat([
                    torch.tensor([1]).to(self.device), mask_chunks[j], torch.tensor([1]).to(self.device)
                ])

                # get required padding length  
                pad_len = max_length - input_id_chunks[j].shape[0]  
    
                # check if tensor length satisfies required chunk size  
                if pad_len > 0:  
                    # if padding length is more than 0, we must add padding  
                    input_id_chunks[j] = torch.cat([  
                        input_id_chunks[j], torch.Tensor([0] * pad_len).to(self.device)
                    ])  
                    mask_chunks[j] = torch.cat([  
                        mask_chunks[j], torch.Tensor([0] * pad_len).to(self.device)  
                    ])  

            stacked_inp_ids = torch.stack(input_id_chunks).to(torch.int64)
            stacked_mask = torch.stack(mask_chunks).to(torch.int64)
            segment_toks = {  
                "input_ids": stacked_inp_ids, 
                "attention_mask": stacked_mask
            }  
            segment_out = self.text_encoder(**segment_toks)
           # print("Pooler output shape: ", segment_out[0][:,0].shape)
            manual_pool_embeddings = self.pool_embeddings(segment_out, segment_toks)
           # print("----Manual Pool: ", manual_pool_embeddings.shape)
            flattened_embed = torch.mean(manual_pool_embeddings, dim=0)
           # print("---Flattened embed: ", flattened_embed.shape)
            list_of_segment_embeddings.append(flattened_embed)  

        batch_embeddings = torch.stack(list_of_segment_embeddings, dim=0)
       # print("Batch of embeddings dims: ", batch_embeddings.shape)
        return batch_embeddings


    def forward(
                self,
                input_ids: torch.LongTensor,
                pixel_values: torch.FloatTensor,
                attention_mask: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None):
            
     #   encoded_text = self.text_encoder(
     #       input_ids=input_ids,
     #       attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
      #      return_dict=True
            
       # )
        #  print("Encoded text: ", encoded_text)
      #  hidden_state = encoded_text[0]
      #  pooler = hidden_state[:, 0]


        
        text_embeddings = self.gen_embeddings_sliding_window(input_ids, attention_mask)
        

        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
        fused_output = self.late_fusion(
            torch.cat(
                [
                   # pooler,
                    text_embeddings,
                    encoded_image.pooler_output,
                ],
                dim=1
            )
        )
        logits = self.pred(fused_output)

        out = {
            "logits": logits
        }

        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out
    
def initialize(config: Dict, answer_space: List[str]) -> MultimodalFusionModel:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultimodalFusionModel(
        config=config,
        num_labels=len(answer_space),
        intermediate_dims=config["model"]["intermediate_dims"],
        dropout=config["model"]["dropout"],
        pretrained_text_model=config["model"]["text_encoder"],
        pretrained_image_model=config["model"]["image_encoder"]
    )
    model.to(device)
    print_gpu_utilization()

    return model 

# COMMAND ----------

# MAGIC %md ### Train with Huggingface Trainer - NOT USED

# COMMAND ----------

import os
import shutil
from typing import Dict, Tuple, List
from transformers import TrainingArguments, Trainer, logging
from datasets import load_metric
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


metric = load_metric('bertscore')

def setTrainingArgs(config: Dict, device) -> TrainingArguments:
    training_args = config["train"]
    if device.type == 'cuda':
        training_args["fp16"] = True
    return TrainingArguments(**training_args)

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
       "bert": metric.compute(predictions=predictions, references=labels),
        "acc": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='macro')
    }


def train(config, device, dataset, collator, model):
    training_args = setTrainingArgs(config, device)
    training_args.output_dir = os.path.join(training_args.output_dir, config["model"]["name"])
    
    if os.path.isdir(training_args.output_dir):
        shutil.rmtree(training_args.output_dir)

    multi_trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    print(multi_trainer.train_dataset[0])

    train_multi_metrics = multi_trainer.train()
    print_gpu_utilization()
    eval_multi_metrics = multi_trainer.evaluate()
    
    return train_multi_metrics, eval_multi_metrics

# COMMAND ----------

import os
import yaml
import logging
import torch 
import transformers
import argparse
from typing import Text
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
# roberta_base

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
    print(os.path.join(config["data"]["data_dir"], config["data"]["train_dataset"]))
    train_ds = Dataset.from_pandas(pd.read_csv(os.path.join(config["data"]["data_dir"], config["data"]["train_dataset"])))
    test_ds = Dataset.from_pandas(pd.read_csv(os.path.join(config["data"]["data_dir"], config["data"]["test_dataset"])))
   # dataset = load_dataset("csv", data_files = {
   #         "train": os.path.join(config["data"]["data_dir"], config["data"]["train_dataset"]),
    #        "test": os.path.join(config["data"]["data_dir"], config["data"]["test_dataset"])
     #   }
   # )
    dataset = {"train": train_ds, "test": test_ds}


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



# if __name__ == '__main__':
   # arg_parser = argparse.ArgumentParser()
    #arg_parser.add_argument('--config', dest='config', required=True)   #need to pass in yaml file 
   # args = arg_parser.parse_args()
  #  main('params.yaml')

# COMMAND ----------

# MAGIC %md ### Finetune model with MLFlow - Current 

# COMMAND ----------

from accelerate import Accelerator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_scalar(name, value, step):
    """Log a scalar value to MLflow"""
    mlflow.log_metric(name, value, step=step)


def acc(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


def train_fn(epoch, config, model, training_loader, accelerator, optimizer, lr_scheduler):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()  #set the model in train mode
    gradient_accumulation_steps = 4
    for batch_idx, data in enumerate(training_loader, 0):
        with accelerator.accumulate(model):
            lens = data['lengths']
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            # ids, lens = pad_packed_sequence(data['input_ids'], batch_first=True)
            ids = [ids[x][:lens[x]] for x in range(len(lens))]
            #  mask, lens = pad_packed_sequence(data['attention_mask'], batch_first=True)
            mask = [mask[x][:lens[x]] for x in range(len(lens))]
            # print("IMAGES: ", data["pixel_values"])
            pixel_values = data["pixel_values"].to(device)
            targets = data["labels"].to(device, dtype=torch.long)
            # print("IDS: ", ids[0].shape)
            # print("PIXELS: ", pixel_values[0].shape)
            # print("Labels: ", targets.shape)
            # token_type_ids = data["token_type_ids"]
            optimizer.zero_grad()
            outputs = model(ids, pixel_values, mask, None, targets)
            loss = outputs["loss"]
            loss = loss #/ gradient_accumulation_steps
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs["logits"], dim=1)
            n_correct += acc(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            #freeze backbone before training 
            if batch_idx % config["train"]["log_interval"] == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy {:.3f}".format(
                        epoch,
                        config["train"]["per_device_train_batch_size"] * batch_idx,
                        len(training_loader.dataset),
                        100.0 * batch_idx / len(training_loader),
                        loss.data.item(),
                        accu_step,
                    )
                )
                # print(f"Training Loss per {} steps: {loss_step}")
                # print(f"Training Accuracy per 250 steps: {accu_step}")
                log_scalar(
                    "train_step_loss", 
                    loss.data.item(),
                    config["train"]["per_device_train_batch_size"] * batch_idx,
                )
            
            accelerator.backward(loss)
            optimizer.step()
           # lr_scheduler.step()
           
            

    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"The Total Accuracy for Epoch {epoch}: {epoch_accu}")
    epoch_loss = tr_loss / nb_tr_steps

    print(f"Training Loss Epoch: {epoch_loss}")
    log_scalar("train_epoch_loss", epoch_loss, epoch)
    log_scalar("train_epoch_acc", epoch_accu, epoch)
    print("----------------------------")


def eval_fn(epoch, model, testing_loader):
    eval_loss = 0
    nb_eval_steps = 0
    n_correct = 0
    nb_eval_examples = 0
    model.eval()
    for batch_idx, data in enumerate(testing_loader, 0):
        ids = data["input_ids"].to(device, dtype=torch.long)
        mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["labels"].to(device, dtype=torch.long)
        lens = data["lengths"]
        ids = [ids[x][:lens[x]] for x in range(len(lens))]
        mask = [mask[x][:lens[x]] for x in range(len(lens))]
        pixel_values = data["pixel_values"].to(device)
     #   token_type_ids = data["token_type_ids"]

        outputs = model(ids, pixel_values, mask, None, targets)
        big_val, big_idx = torch.max(outputs["logits"], dim=1)
        n_correct += acc(big_idx, targets)
        loss = outputs["loss"]
        eval_loss += loss.item()
        nb_eval_steps += 1
        nb_eval_examples += targets.size(0)
    epoch_loss = eval_loss / nb_eval_steps
    epoch_accu = (n_correct * 100) / nb_eval_examples
    print(f"The Total Accuracy for Epoch {epoch}: {epoch_accu}")
    print(f"Testing Loss Epoch: {epoch_loss}")
    log_scalar("eval_epoch_loss", epoch_loss, epoch)
    log_scalar("eval_epoch_acc", epoch_accu, epoch)
    print("----------------------------")
    return epoch_accu

# COMMAND ----------

import mlflow

class LMWrapper(mlflow.pyfunc.PythonModel):
    """
    Creates a wrapper for an MLFlow Python model that contains custom inference logic
    Allows programmatic saving/logging of attributes and artifacts of a particular run
    """
    def __init__(self, model, tokenizer, preprocessor, mapping_dict, max_len):
        self.model = model
        self.tokenizer = tokenizer
        self.img_preprocessor = preprocessor
        # maps category (number) to class
        self.mapping_dict = mapping_dict
        self.max_len = max_len
        self.transform = transforms.Resize(124)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict(self, context, model_input):
        """
        Custom prediction function

        Parameters:
            model_input (text)
        """
        model_input_txt = model_input["text"]
        model_input_img = model_input["image"]
        resized_img = transforms.ToTensor()(self.transform(Image.open(io.BytesIO(model_input_img)).convert('RGB')))
        processed_img = self.img_preprocessor(images=[resized_img], return_tensors="pt")
        tokens = {"input_ids": [], "attention_mask": []}
        new_tokens = self.tokenizer(
            model_input_txt,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
        )
        tokens["input_ids"].append(new_tokens["input_ids"][0])
        tokens["attention_mask"].append(new_tokens["attention_mask"][0])
        # reformat list of tensors into single tensor
        tokens["input_ids"] = torch.stack(tokens["input_ids"])
        tokens["attention_mask"] = torch.stack(tokens["attention_mask"])
        if self.device.type == "cuda":
            tokens["input_ids"] = tokens["input_ids"].to("cuda")
            tokens["attention_mask"] = tokens["attention_mask"].to("cuda")
        outputs = self.model(**tokens, pixel_values=processed_img['pixel_values'].squeeze())
        _, big_idx = torch.max(outputs.data, dim=1)
        ret_d = {
            "class": np.array(
                [self.mapping_dict[str(x)] for x in big_idx.cpu().numpy()]
            )
        }
        return ret_d


def log_classifier_model(model, tokenizer, preprocessor, encode_dict, model_params):
    """
    :param model: A model that takes as input text and outputs a category (for the question)
    :param tokenizer: The tokenizer associated with the model
    :param encode_dict: A dictionary which encodes a labelled category (such as 'Guest Rooms') to a numeric category
    :param model_params: Paremeters used to set up the model and its training
    """
    mapping_dict = {str(v): k for k, v in encode_dict.items()}
    wrapped_model = LMWrapper(model, tokenizer, preprocessor, mapping_dict, model_params["MAX_LEN"])
    input_example = {
        "text": np.array([])
    }
    signature = infer_signature(
        pd.DataFrame(input_example),
        wrapped_model.predict(None, pd.DataFrame(input_example)),
    )
    # Specify the additional dependencies
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "torch=={}".format(torch.__version__),
            "transformers=={}".format(transformers.__version__),
        ],
        additional_conda_channels=None,
    )
    # Log model so the model artifacts can have the correct conda env, python model, and signature (data formats for input & output)
    return mlflow.pyfunc.log_model(
        "Auto Result Analysis - BERT",
        python_model=wrapped_model,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
    )

# COMMAND ----------

from accelerate import Accelerator
from mlflow.tracking import MlflowClient
import mlflow

client = MlflowClient()
def init_experiment(experiment_name):
    # fetch existing experiments, if it doesn't exist create it and set it
    existing_experiments = client.search_experiments()
    if experiment_name in [exp.name for exp in existing_experiments]:
        return mlflow.set_experiment(experiment_name).experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
        # artifact_location = "/Users/Chris.Jose@cvent.com/test_result_analysis/experiments")

# COMMAND ----------

import pandas as pd
import shutil
from typing import Dict
from transformers import get_linear_schedule_with_warmup
from collections import OrderedDict


encode_dict_ = {}

def encode_label(x):
    if x not in encode_dict_.keys():
        encode_dict_[x] = len(encode_dict_)
    return encode_dict_[x]


def prep_data(config: Dict):
    df = pd.read_parquet(os.path.join(config["data"]["data_dir"], config["data"]["full_dataset"]))
    df.dropna(subset=['screenshot', 'IssueType'], inplace=True)
    df['image_id'] = f'img_{df.ExecutionId}_{df.ExecutionItemID}_{df.StepLogID}'
    keys_to_remove = ['TEST DATA', 'DB DATA', 'PLATFORM', 'CAPABILITIES', 'RESOLUTION']  
    df.reset_index(drop=True, inplace=True)
    cudf_df = cudf.from_pandas(df)
    cudf_df["StepDescription"] = cudf_df["StepDescription"].applymap(clean_and_parse, keys_to_remove)
    cudf_df["CAT"] = cudf_df["IssueType"].applymap(lambda x: encode(label(x)))
    #for i, text in enumerate(df["StepDescription"]):
    #    for key in keys_to_remove:  
    #        text = re.sub(rf"{key}:.*?\n", "", text)  
    #    df.at[i, "StepDescription"] = text
    
    #df["CAT"] = df["IssueType"].map(lambda x: encode_label(x))
    df = cudf_df.to_pandas()
    df_reduc = df[['StepDescription', 'image_id', 'screenshot', 'IssueType', 'CAT']]
    labels = df_reduc["IssueType"].unique()
    return df_reduc, labels

def train_classifier(config):
    """
    Wraps model training and experiment logging into one method.

    :param model_name: Model to train. Currently one of ACCEPTED_MODELS
    :param manual_pooling: If base model is to be frozen and only a projection head is trained (e.g distilbert_cvent approach), then specifies
        whether the output embeddings should be pooled into a single embedding (if True) OR to only use the embeddings for the CLS tokens (if False)
    :param df: Pandas dataframe to use for training, testing, and model eval. Must contain question and category column.
    :param model_params: Dictionary of model parameters to train/evaluate the model.
    """
    mlflow.end_run()
    #experiment_id = init_experiment("dbfs:/FileStore/shared_uploads/Chris.Jose@cvent.com/test_result_analysis/experiments")

    experiment_id = init_experiment("/Users/Chris.Jose@cvent.com/test_result_analysis/experiments/test_result_analysis_2")
    #experiment = mlflow.get_experiment(experiment_id)
   
    run = mlflow.start_run(run_name="distilbert_base_classifier", experiment_id=experiment_id)


    # get dataset and dataloadets
    df, labels = prep_data(config)
    print(df.head())
    (training_loader, testing_loader, training_set, testing_set, tokenizer, preprocessor) = createMultimodalDataCollator(config, df)
    
  # define the model
    model = initialize(config, labels)

    # Creating the loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config["train"]["learning_rate"]
    )

    gradient_accumulation_steps = 4

    lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=(len(training_loader) * config["train"]["num_train_epochs"]) // gradient_accumulation_steps,
    )
  
    accelerator = Accelerator(mixed_precision = 'fp16')

   # model.gradient_checkpointing_enable()
    
    model, optimizer, training_loader, testing_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, training_loader, testing_loader, lr_scheduler
    )

    # Log the parameters used to train the model (such as learning rate, epochs, etc)
    for key, value in config["train"].items():
        mlflow.log_param(key, value)

    if config["train"]["resume"]:
        if os.path.isfile(config["train"]["resume"]):
            print("=> loading checkpoint '{}'".format(config["train"]["resume"]))
            ckpt = torch.load(config["train"]["resume"])
            start_epoch = ckpt["epoch"]
            state_dict = ckpt['model']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)


           # new_state_dict = OrderedDict()
           # for k, v in state_dict.items():
           #     if 'module' not in k:
           #         k = 'module.'+k
           #     else:
           #         k = k.replace('features.module.', 'module.features.')
           #     new_state_dict[k]=v
           # model.load_state_dict(new_state_dict)
        else:
            print("=> no checkpoint found at '{}'".format(config["train"]["resume"]))
    else:
        start_epoch = 0

    best_accu = 0
    os.makedirs(config["train"]["checkpoints"], exist_ok=True)
    for epoch in range(start_epoch, config["train"]["num_train_epochs"]):
        print("Active Run ID: %s, Epoch: %s \n" % (run.info.run_uuid, epoch))
        # Perform training and evaluation. NOTE: Not shown, but each call to train/test logs metrics to MlFlow
        train_fn(epoch, config, model, training_loader, accelerator, optimizer, lr_scheduler)
        epoch_accu = eval_fn(epoch, model, testing_loader)
        is_best = epoch_accu > best_accu
        best_accu = max (epoch_accu, best_accu)
        accelerator.wait_for_everyone()
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict()
        }, config["train"]["checkpoints"] + 'checkpoint_{}.pth.tar'.format(epoch))
        if is_best:
            shutil.copyfile(config["train"]["checkpoints"] + 'checkpoint_{}.pth.tar'.format(epoch), config["train"]["checkpoints"] + 'model_best.pth.tar')
                
    return model, tokenizer, testing_set

# COMMAND ----------

import yaml
import torch
import os
CONF_PATH = "params.yaml"
with open(CONF_PATH) as conf_f:
    config = yaml.safe_load(conf_f)
    
if config["base"]["use_cuda"]:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # SET ONLY 1 GPU DEVICE
else:
    device =  torch.device('cpu')
tmp_model, tokenizer, testing_set = train_classifier(config)

# COMMAND ----------

def compute_confusion_matrix(y_true, y_pred, title, encode_dict, is_percentage=False):
    font = {"family": "normal", "size": 15}
    matplotlib.rc("font", **font)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    if not is_percentage:
        sns.heatmap(
            cm,
            annot=True,
            annot_kws={"size": 12},
            ax=ax,
            cmap="Blues",
            fmt="g",
            xticklabels=list(encode_dict.keys()),
            yticklabels=list(encode_dict.keys()),
        )
    else:
        sns.heatmap(
            cm / cm.sum(axis=1).reshape(-1, 1),
            annot=True,
            annot_kws={"size": 12},
            ax=ax,
            cmap="Blues",
            fmt=".2%",
            xticklabels=list(encode_dict.keys()),
            yticklabels=list(encode_dict.keys()),
        )
    plt.xticks(rotation=45, horizontalalignment="right")
    fig.set_size_inches(12, 12)
    plt.title(title, fontdict={"family": "normal", "size": 21})
    img_buf = io.BytesIO()
    plt.ylabel("True label", fontdict={"family": "normal", "size": 21})
    plt.xlabel("Predicted label", fontdict={"family": "normal", "size": 21})
    plt.show()
    fig.savefig(img_buf, dpi=fig.dpi, bbox_inches="tight")
    return img_buf


def eval_classifier(model, testing_set, encode_dict, log: bool = True):
    """
    Take a trained model and evaluate it by calculating metrics + confusion matrices. Log these outputs to the mlflow run

    :param model: A trained pytorch language model to evaluate
    :param testing_set: A torch Dataset object that contains tokenized data to evaluate
    :param encode_dict: Dictionary that maps a categorical name to a categorical index (class name to number)
    :param log: Boolean on whether to log results to MlFlow
    """
    # use test set for it
    model.eval()
    y_true = []
    y_pred = []
    for idx in range(len(testing_set)):
        data = testing_set[idx]
        y_true.append(data["labels"].item())
        ids = torch.unsqueeze(data["ids"], 0).to("cuda")
        mask = torch.unsqueeze(data["mask"], 0).to("cuda")
        pixels_values = torch.unsqueeze(data["pixel_values"], 0).to("cuda")
        out = torch.max(model(ids, pixel_values, mask), 1).indices.cpu().numpy()[0]
        y_pred.append(out)

    # compute confusion matrices
    cm_buf = compute_confusion_matrix(
        y_true, y_pred, "Confusion Matrix - Test Set", encode_dict, is_percentage=False
    )
    cm_buf_percentage = compute_confusion_matrix(
        y_true,
        y_pred,
        "Confusion Matrix - Test Set Percentage",
        encode_dict,
        is_percentage=True,
    )
    if log:
        im = Image.open(cm_buf)
        mlflow.log_image(im, "confusion_matrix.png")
        cm_buf.close()
        # Now do percentage matrix
        im = Image.open(cm_buf_percentage)
        mlflow.log_image(im, "confusion_matrix_percentage.png")
        cm_buf_percentage.close()

    # Compute F1, prec, rec
    how = ["micro", "macro", "weighted"]
    metrics = {k: {"f1": 0, "precision": 0, "recall": 0} for k in how}
    for k in how:
        metrics[k]["f1"] = f1_score(y_true, y_pred, average=k)
        metrics[k]["precision"] = precision_score(y_true, y_pred, average=k)
        metrics[k]["recall"] = recall_score(y_true, y_pred, average=k)
    # now print it nicely
    print("Metrics for test set!")
    print("+" * 29)
    for k in how:
        print(f"{k.upper()} average:")
        for k2 in metrics[k]:
            print(f"\t{k2}: {metrics[k][k2]}")
        print("-" * 29)
    if log:
        # log f1,prec,recall for micro, macro, and weighted (so 9 total)
        for k in how:
            for k2 in metrics[k]:
                mlflow.log_metric(f"{k}_{k2}", metrics[k][k2])

# COMMAND ----------

import os
os.makedirs(config["train"]["checkpoints"], exist_ok=True)
os.path.exists('/Workspace/Repos/Chris.Jose@cvent.com/auto-result-analysis/checkpoints/')

# COMMAND ----------

os.makedirs(config["metrics"]["metrics_folder"], exist_ok=True)
metrics_path = os.path.join(config["metrics"]["metrics_folder"], config["metrics"]["metrics_file"])

# COMMAND ----------

# MAGIC %pwd

# COMMAND ----------

print_gpu_utilization()

# COMMAND ----------

# MAGIC %sh python /Workspace/Repos/Chris.Jose@cvent.com/auto-result-analysis/cuda_profile.py > out.txt

# COMMAND ----------

torch.cuda.memory_summary(device=None, abbreviated=False) 

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

import torch
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)


# COMMAND ----------

# MAGIC %md ### Deployment 

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

base:
  use_cuda: True

data:
  data_dir: /dbfs/mnt/s3_mount/qe_automation/data
 # data_dir: /dbfs/FileStore/shared_uploads/Chris.Jose@cvent.com
  images_folder: images
  full_dataset: data_optimized_proc.parquet
#  full_dataset: subset.csv 
  train_dataset: train_dataset.csv
  test_dataset: test_dataset.csv
  description_col: StepDescription
  image_col: image_id
  answer_col: IssueType
  answer_space: answer_space.txt
  train_size: 


tokenizer:
  padding: True
  max_length: 512
  truncation: True
  add_special_tokens: True
  return_token_type_ids: True
  return_attention_mask: True


model:
  name: roberta-beit  # Custom name for the multimodal model
  text_encoder: vives/distilbert-base-uncased-finetuned-cvent-2019_2022
  image_encoder: microsoft/beit-base-patch16-224-pt22k-ft22k  
  intermediate_dims: 512
  dropout: 0.5

train:
  output_dir: mlflow_checkpoint
  path: /Users/Chris.Jose@cvent.com/test_result_analysis/experiments/
  checkpoints: /dbfs/mnt/s3_mount/qe_automation/checkpoints/
  resume: False 
    #/dbfs/mnt/s3_mount/qe_automation/checkpoints/model_best.pth.tar
  seed: 12345
  num_train_epochs: 50
  learning_rate: 5.0e-5
  weight_decay: 0.0
  warmup_ratio: 0.0
  warmup_steps: 0
  evaluation_strategy: steps
  eval_steps: 100
  log_interval: 10
  logging_strategy: steps
  logging_steps: 100
  save_strategy: steps
  save_steps: 100
  save_total_limit: 3            # Save only the last 3 checkpoints at any given time while training 
  metric_for_best_model: wups
  per_device_train_batch_size: 128
  per_device_eval_batch_size: 128
  remove_unused_columns: False
  dataloader_num_workers: 8
  load_best_model_at_end: True

metrics:
  metrics_folder: /dbfs/FileStore/shared_uploads/Chris.Jose@cvent.com/metrics
  metrics_file: metrics.json

inference:
  checkpoint: checkpoint-1500
