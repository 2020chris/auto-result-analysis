# Databricks notebook source
# MAGIC %md # XGBoost on TF-IDF Features

# COMMAND ----------

# MAGIC %pip install openpyxl

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    f1_score,
    recall_score,
    precision_score,
)
from xgboost import XGBClassifier
import xgboost
import scipy.stats as st
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.models.signature import infer_signature
import matplotlib
import io
from PIL import Image
import numpy as np
import seaborn as sns
import sklearn

# COMMAND ----------

# MAGIC %md # Prepare Data

# COMMAND ----------

bucket_name = "cvent-databricks-root-storage-pr50"
mount_name = "s3-mount"
mount_point = "/mnt/%s" % mount_name
try:
    dbutils.fs.mount("s3a://%s" % bucket_name, mount_point)
except Exception as e:
    print("Already mounted!")

# COMMAND ----------

# Load and prepare data
encode_dict = {}


def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]


df = pd.read_excel(
    
"/dbfs/mnt/s3-mount/csn_rfp_question_standard/CSN_top100accts_questionsWithCategoryLabels_Feb2023_processed.xlsx"
)
df = df.dropna(subset=["category"])
df["CAT"] = df["category"].map(lambda x: encode_cat(x))
print(df.groupby(["CAT", "category"])["CAT"].count())
print(df.groupby(["CAT", "category"])["CAT"].count() / df.shape[0])
df.head()

# COMMAND ----------

train_dataset, test_dataset = train_test_split(
    df, train_size=0.8, random_state=42, stratify=df.CAT
)
train_dataset = train_dataset.reset_index(drop=True)
test_dataset = test_dataset.reset_index(drop=True)
print("- TRAIN -\n")
print(train_dataset.groupby(["CAT", "category"])["CAT"].count())
print("-----------------")
print("- TEST -\n")
print(test_dataset.groupby(["CAT", "category"])["CAT"].count())

# COMMAND ----------

# MAGIC %md # Set CV Params
# MAGIC 
# MAGIC Hyperparam ranges cloned from Job Classifier

# COMMAND ----------

def get_params():
    learning_rate = st.uniform(0.05, 0.4)
    max_depth = [10, 20, 50, 100, 200, 300, 500]
    n_estimators = [5, 10, 25, 50, 75, 100, 200, 400]
    reg_lambda = [0.1, 0.3, 1]
    colsample_bytree = st.beta(10, 1)
    gamma = st.uniform(0, 10)
    subsample = [0.6, 0.8, 1.0]
    reg_alpha = st.expon(0, 50)
    min_child_weight = [1, 5, 10]
    # Create the random grid
    return {
        "vectorizer__analyzer": ["word"],
        "vectorizer__stop_words": ["english"],
        "vectorizer__strip_accents": ["ascii"],
        "vectorizer__ngram_range": [(1, 1), (1, 2), (2, 2)],
        "classifier__n_estimators": n_estimators,
        "classifier__learning_rate": learning_rate,
        "classifier__reg_lambda": reg_lambda,
        "classifier__colsample_bytree": colsample_bytree,
        "classifier__gamma": gamma,
        "classifier__subsample": subsample,
        "classifier__reg_alpha": reg_alpha,
        "classifier__max_depth": max_depth,
        "classifier__min_child_weight": min_child_weight,
    }

# COMMAND ----------

# MAGIC %md # Define model

# COMMAND ----------

def train_tree_model(
    train_data: pd.DataFrame, cv_params: dict, n_cv_folds: int, n_iter: int
):
    """
    :param train_data: Preprocessed dataframe that contains both the question and category
    """
    X = train_data["question"]
    Y = train_data["CAT"]
    classifier = Pipeline(
        steps=[
            ("vectorizer", TfidfVectorizer()),
            ("classifier", XGBClassifier(random_state=10)),
        ],
        verbose=True,
    )
    classifier_random = RandomizedSearchCV(
        classifier,
        param_distributions=cv_params,
        n_iter=n_iter,
        verbose=0,
        scoring="accuracy",
        random_state=42,
        refit=True,
        n_jobs=-1,
        cv=StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42),
    )
    classifier_random.fit(X, Y)
    return classifier_random

# COMMAND ----------

# MAGIC %md ## Model storage function

# COMMAND ----------

class SKLearnEstimatorWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, estimator, mapping_dict):
        self.estimator = estimator
        self.mapping_dict = mapping_dict

    def predict(self, context, model_input):
        model_input = model_input["text"]
        outputs = self.estimator.predict(model_input)
        ret_d = {"class": np.array([self.mapping_dict[str(x)] for x in outputs])}
        return ret_d


def log_estimator_model(estimator, encode_dict):
    """
    :param estimator: An estimator that takes as input text and outputs a category (for the 
question)
    :param encode_dict: A dictionary which encodes a labelled category (such as 'Guest Rooms') to a 
numeric category
    """
    mapping_dict = {str(v): k for k, v in encode_dict.items()}
    wrapped_estimator = SKLearnEstimatorWrapper(estimator, mapping_dict)
    input_example = {
        "text": np.array(
            [
                "Number of sleeping rooms",
                "COVID-19 isolation requirements",
                "fees for cleaning",
            ]
        )
    }
    signature = infer_signature(
        pd.DataFrame(input_example),
        wrapped_estimator.predict(None, pd.DataFrame(input_example)),
    )
    # Specify the additional dependencies
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "xgboost=={}".format(xgboost.__version__),
            "sklearn=={}".format(sklearn.__version__),
        ],
        additional_conda_channels=None,
    )
    # Log model so the model artifacts can have the correct conda env, python model, and signature 
(data formats for input & output)
    return mlflow.pyfunc.log_model(
        "CSN_question_classifier",
        python_model=wrapped_estimator,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
    )

# COMMAND ----------

# MAGIC %md # Perform CV with MlFlow

# COMMAND ----------

client = MlflowClient()
n_cv_folds = 3


def init_experiment(experiment_name):
    # fetch existing experiments, if it doesn't exist create it and set it
    existing_experiments = client.search_experiments()
    if experiment_name in [exp.name for exp in existing_experiments]:
        return mlflow.set_experiment(experiment_name)
    else:
        return mlflow.create_experiment(experiment_name)

# COMMAND ----------

mlflow.end_run()
experiment_id = init_experiment(
    
"/Users/CVives@cvent.com/CSN_RFP_question_standardization/experiments/question_classification_experiment"
).experiment_id
run = mlflow.start_run(run_name="xgboost_tfidf_CV", experiment_id=experiment_id)
mlflow.sklearn.autolog(log_models=False)
mlflow.log_param("n_cv_folds", 3)
classifier_random = train_tree_model(
    train_dataset, get_params(), n_cv_folds, n_iter=30000
)
print(classifier_random.best_score_)
X_test = test_dataset["question"]
Y_test = test_dataset["CAT"]
mlflow.log_metric(
    "testing_accuracy_score", classifier_random.best_estimator_.score(X_test, Y_test)
)
estimator = classifier_random.best_estimator_

# COMMAND ----------

# MAGIC %md # Test set metrics

# COMMAND ----------

y_pred = classifier_random.best_estimator_.predict(X_test)
y_true = Y_test

# COMMAND ----------

# MAGIC %md ## Confusion matrices

# COMMAND ----------

font = {"family": "normal", "size": 15}

matplotlib.rc("font", **font)
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots()
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
plt.xticks(rotation=45, horizontalalignment="right")
fig.set_size_inches(12, 12)
plt.title("Confusion Matrix - Test Set", fontdict={"family": "normal", "size": 21})
img_buf = io.BytesIO()
plt.ylabel("True label", fontdict={"family": "normal", "size": 21})
plt.xlabel("Predicted label", fontdict={"family": "normal", "size": 21})
plt.show()
fig.savefig(img_buf, dpi=fig.dpi, bbox_inches="tight")

# COMMAND ----------

font = {"family": "normal", "size": 15}

matplotlib.rc("font", **font)
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots()
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
plt.title(
    "Confusion Matrix - Test Set Percentage", fontdict={"family": "normal", "size": 21}
)
plt.ylabel("True label", fontdict={"family": "normal", "size": 21})
plt.xlabel("Predicted label", fontdict={"family": "normal", "size": 21})
img_buf2 = io.BytesIO()
plt.show()
fig.savefig(img_buf2, dpi=fig.dpi, bbox_inches="tight")

# COMMAND ----------

# MAGIC %md ## F1, precision, recall

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md # Log confusion matrix + metrics

# COMMAND ----------

im = Image.open(img_buf)
mlflow.log_image(im, "confusion_matrix.png")
img_buf.close()
# Now do percentage matrix
im = Image.open(img_buf2)
mlflow.log_image(im, "confusion_matrix_percentage.png")
img_buf.close()

# COMMAND ----------

# log f1,prec,recall for micro, macro, and weighted (so 9 total)
for k in how:
    for k2 in metrics[k]:
        mlflow.log_metric(f"{k}_{k2}", metrics[k][k2])

# COMMAND ----------

# MAGIC %md # Log model to run

# COMMAND ----------

log_estimator_model(estimator, encode_dict)

# COMMAND ----------

# MAGIC %md # End run

# COMMAND ----------

mlflow.end_run()
