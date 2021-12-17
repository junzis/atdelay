# Air-traffic-delays-prediction-model

  
This project is part of the Capstone Project (TI3150TU) of the minor Engineering with AI at Delft University of Technology.

## Contributors
The following people contributed to the project:

* Tristan Dijkstra
* Niels Prins
* Constantinos Aristodemou
* Benjamin Slijper
* Theodor Falat
* Vlad Buzetelu
* Tim Hogenelst

## Project description
The aim of this project is to build a model which can predict airport traffic delay propagation. This will be done using both flight plans and actual flight trajectories collected over the entire Europe between 2015 and 2019. See [**Data**](#data) for more information.

This project consists of two parts:

1. A machine learning model to predict delays at single airports;
2. A neural network to predict the propagation of delays through the air traffic network.

#### Single airport prediction
A Random Forest regression model was used to obtain delays at individual airports. Features such as airline, planned arrival time and airport capacity were used as input to predict the target variable, which is *arrival delay*. 

#### Delay propagation
some summary about delay propagation


## Installation steps
This package requires the following other programs or packages to be installed:
* add prerequisite packages (and link to install if applicable?)

You can install the package by running the following command in the command prompt:
```
pip install Air-traffic-delays-prediction-model or something like this
```


## How to use
description of workflow (something to do with the notebooks maybe? or some other kind of code example)

## Data 
An overview of all of the various data sources used for feature engineering:

* Flight data was taken from the EUROCONTROL R&D Archive
* capacity data
* top 50 airport data
