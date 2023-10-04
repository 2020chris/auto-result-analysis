# auto-result-analysis
Automate Blucumber test result analysis with ML


# File structure
fetch_data.py - script to fetch steplogs from mysql qe_automation db

bert-result-analysis.ipynb.py - Databricks notebook to fine tune a neural network on top of distilbert model (or can fine tune distilbert model) 
for text classification task. Import into Databricks workspace and rerun notebook cells sequentially.

bert-vit-result-analysis.ipynb.py - Databricks notebook to fine tune a neural network on top of a frozen distilbert model for a text + image classification task. Import into Databricks workspace and rerun notebook cells sequentially.

params.yaml - Contains all necessary input parameters

xgboost-*.ipynb - xgboost classifier on encoded tabular data
