# data_engineering_figure8
Data Engineering with Figure 8 disaster response tweets

## Installations 
The following code is based on Python 3 and contains the following packages:  json, pandas, numpy, nltk, sqlalchemy, re, sklearn, pickle, sys

## Overview
This repository contains an Extract-Transform-Load (ETL) pipeline and a Machine learning (ML) pipeline to classify tweets from natural disasters into different categories that could be used during a natural emergency. This was done with data [from Figure8](https://www.figure-eight.com/). 

## Instructions
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/disaster_model.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/

## File Descriptions

> App
>> run.py : runs main functions to display on webpage
>> templates: web templates


> Data 
>> DisasterRespnse.py : cleaned database after ETL pipeline
>> disaster_categories.csv : raw data of categories to be used
>> disaster_messages.csv : csv file with input disaster messages
>> process_data.py : ETL pipeline

> Models
>> train_classifer.py : Machine Learning pipelien

> Notebooks
>> Jupyter notebooks
