# auto-result-analysis - AUTO-65515
Automate Blucumber test result analysis with ML
https://wiki.cvent.com/pages/viewpage.action?pageId=573872491

# File structure
fetch_data.py - script to fetch steplogs from mysql qe_automation db
bert-result-analysis.ipynb.py - Databricks notebook to fine tune a neural network on top of distilbert_cvent model (or can fine tune distilbert model) for text classification task
bert-vit-result-analysis.ipynb.py - Databricks notebook to fine tune a neural network on top of a frozen distilbert model for a text + image classification task
params.yaml - Contains all necessary input parameters
xgboost-*.ipynb - xgboost classifier on encoded tabular data