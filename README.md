# Air-traffic-delays-prediction-model

  
This project was made as part of the Capstone Project (TI3150TU) of the minor Engineering with AI at Delft University of Technology. (Dec 2021 - Jan 2022)  

## Contributors
The following people contributed to the project:

* Constantinos Aristodemou @ConstantinosAr
* Vlad Buzetelu @vladbuzetelu
* Tristan Dijkstra @IrTrez
* Theodor Falat @theofalat
* Tim Hogenelst @TimGioHog
* Niels Prins @Niels-Prins 
* Benjamin Slijper @BenjaminSlijper


## Project description
The aim of this project was to build Machine learning models which can predict airport traffic delays and their propagations in European airports. Flight data was retrieved from EUROCONTROL with additional data found elsewhere. (See [**Data**](#acquiring-the-data) for more information.)

The project resulted in 3 different models:

1. A Random Forest model that predicts delays for individual flights
2. An LSTM model that predicts time-aggregated delays for a single airport.
3. An ST-GNN (Spatial Temporal Graph Neural Network) model that predicts delays for multiple airports at the same time.

The models are showcased in their respective jupyter notebooks.

Additionally, supporting functions have been created to process the data and visualise it. Most of the central data processing is done in the extraction module.

## Installation steps
### Note on documentation
Please note that **all** functions in the repository include an in-depth docstring that explains the function and its parameters. To showcase the work done, jupyter notebooks have been created, explanation on the processes to achieve results are explained there.
### Prerequisite packages
This project requires a significant amount of packages to work, including tensorflow. A lot of time was spend getting these to work properly. Ultimately, in our experience the easiest way to install it on windows was using pip and not conda. We recommend creating a virtual enviroment via pip and installing the packages via our requirements.txt file using:
```
pip install -r requirements.txt
```
### Configuring Tensorflow GPU functionality. (Optional)
Training neural networks and in particular Graph Neural Networks tends to be quite slow. We therefore also recommend configuring tensorflow to work with a graphics card (CUDA compatible NVIDIA GPU required). To do this some additional dependancies should be installed (see Tensorflow GPU guide). Importantly, all the versions should be compatible. Figuring out which versions to install can be tricky. At the time of submission, the following versions were used and should be compatible:

- Windows 10
- Tensorflow 2.7.0
- CUDA Toolkit 11.2
- cudnn 8.1.0

Details on where to find and how to install the the dependencies can be found here:

- [Tensorflow GPU guide](https://www.tensorflow.org/install/gpu)

- [cudnn install guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)


TODO:

You can install the package by running the following command in the command prompt:
```
pip install Air-traffic-delays-prediction-model or something like this
```


### Acquiring the data 
The project uses 3 main sets of data:
- Flights data, provided by EUROCONTROL - (TODO ADD LINK FOR GENERAL AUDIENCE) (For graders of the capstone project this is provided in the readme.txt)
- Weather data provided by [NCEI](https://www.ncei.noaa.gov/) - retrieved by the programme automatically.
- Airport information (Coordinates etc) - provided in the repository for the Europe's top 50 airports.

The usage of each of these is summarised in the chart below. Many of the functions in extract have a 'generate/write/read' capability meaning they generate the full data on the first cold run and return the stored filtered data from the data on subsequent runs. This is indicated by the hollow arrows in the chart below. Some of the functions can take long to generate, for example generating the weather data and Neural Network data for the top 50 airports can take up to an hour each. For capstone graders, reduced versions of the the filtered datasets are provided in the readme.txt. For potential legal reasons they are not provided publically in this repository.

![function chart of extraction](/docs/funcChart.png)


#### File structure
Your file tree should look something like this:
```
Air-traffic-delays-prediction-model
├───data
│   ├───2018
│   │   ├───201803
│   │   ├───201806
│   │   ├───201809
│   │   └───201812
│   ├───2019
│   │   ├───...
.   .   .
.   .
.   .
│   └───Weather_Data_Filtered
│       ├───...
.   
.   
.   
├───filteredData
├───LRData
├───NNData
.
.
.
```

## Extraction

TODO TALK ABOUT SOME OF THE FUNCTIONS IN THE FLOW CHART

## Models
### Individual flight prediction
A Random Forest regression model was used to obtain delays at individual airports. Features such as airline, planned arrival time and airport capacity were used as input to predict the target variable, which is *arrival delay*. 

TODO EDIT THIS SECTION

### Single airport prediction
TODO EDIT THIS SECTION

### Graph Neural Network

TODO EDIT THIS SECTION

## Troubleshooting and Contact
TODO MAYBE OR MAYBE NOT ADD THIS