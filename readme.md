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