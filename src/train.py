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

    train_multi_metrics = multi_trainer.train()
    eval_multi_metrics = multi_trainer.evaluate()
    
    return train_multi_metrics, eval_multi_metrics