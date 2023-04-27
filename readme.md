# AirBnB Predictions

## Introduction
This project have be develop on data collected from [InsideAirBnB](http://insideairbnb.com/get-the-data/), that are separated in listings and reviews (comments on the listings).

## Folder structure

* **data_analysis**: contains notebooks with the analysis of the data.
* **dataset**: contains the data and some custom classes to work with them.
    * *listings*: contains the data relative the listings.
    * *comments*: contains the data relative the listings's reviews of comments.
* **embeddings**: contains the processed data in **pikles** that we obtain in the intermediate seteps of the data_processing.
* **model/models**: contains custom NeuralNetowoks model developed for the project.
* **processing**: contains notebooks used to preproces the data cleaning them or generate embeddings using Sentence Models.
* **utils**: contains varius utils to process the data, specal note for the amenities a special field present in the listing processed using clusters.
* **visualization**: contains custom modules to visualize the data.

## Running

### 0. Requirments 
This code have been developed on python 3.11.3, we recomend an equal mayor related version

### 1. Install the eviorment

Eviorment setup
1. Using Pip

```bash
pip install -r requirements.txt
```

2. Using Conda
```bash
conda env create -f environment.yml
```

### 3. Dataset - SetUp

We need to placed the dataset downloaded from [InsideAirBnB](http://insideairbnb.com/get-the-data/), inside the dataset folders.
* The ***lising.csv*** have to be placed inside *dataset/lisining* folder, you can place more than one all the csv files in the folder will be used.
* The ***reviews.csv*** have to be placed inside *dataset/comments* folder, you can place more than one all the csv files in the folder will be used.


### 4. Data Processing
Run in order the processing steps:
1.  ***step1_merge_listings_comments.ipynb***: connects the listings and reviews togheter.
2.  ***step2_process_columns.ipynb***: generates the embeddings for the commnets-review for the listings.
3.  ***step3_process_comments.ipynb***: generates the embeddinigs, as well as, the processed ordinal and numeric data.
E.G., prices or listing type (Apartment, Home, etc...).
4.  ***step4_extraction_of_test_set.ipynb***: merge the embeddings and pre-processed data and generate the train and test dataset files.

### 5. Data Analysis 

Simply open any notebook in the data_analysis and run it, *only remeber that the analysis require the embeddings to be computed*.
As such you need the data pre-processing first.

### 6. Experiments

You can simply run any notebook to evaluate our models on the the data provided the experiments are separated as floow: