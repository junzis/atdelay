# Air Traffic Delays Prediction Model

  
This project was made as part of the Capstone Project (TI3150TU) of the minor Engineering with AI at Delft University of Technology. 

Dec 2021 - Jan 2022

## Contributors
The following people contributed to the project as part of Capstone:

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
Please retrieve the repository through git to run it:
```
git clone https://github.com/ConstantinosAr/Air-traffic-delays-prediction-model.git
```
### Note on documentation
Please note that **all** functions in the repository include an in-depth docstring that explains the function and its parameters. To showcase the work done, jupyter notebooks have been created, explanation on the processes to achieve results are explained there.
### Prerequisite packages
This project requires a significant amount of packages to work, including tensorflow. A lot of time was spend getting these to work properly. Ultimately, in our experience the easiest way to install it on windows was using pip and not conda. We recommend creating a virtual enviroment via pip and installing the packages via our requirements.txt file using:
```
pip install -r requirements.txt
```

#### Manual installation
If you would like to install the packages manually it can be done with:
```
pip install matplotlib networkx numpy pandas requests scikit_learn scipy seaborn spektral tensorflow tqdm xarray
```
The project has been tested to work with the following versions of these libraries:
- matplotlib>=3.4.2
- networkx>=2.6.3
- numpy>=1.22.1
- pandas>=1.3.3
- requests>=2.23.0
- scikit_learn>=1.0.2
- scipy>=1.5.0
- seaborn>=0.11.1
- spektral>=1.0.8
- tensorflow>=2.7.0
- tqdm>=4.46.0
- xarray>=0.20.2
### Configuring Tensorflow GPU functionality. (Optional)
Training neural networks and in particular Graph Neural Networks tends to be quite slow. We therefore also recommend configuring tensorflow to work with a graphics card (CUDA compatible NVIDIA GPU required). To do this some additional dependancies should be installed (see Tensorflow GPU guide). Importantly, all the versions should be compatible. Figuring out which versions to install can be tricky. At the time of submission, the following versions were used and should be compatible:

- Windows 10
- Tensorflow 2.7.0
- CUDA Toolkit 11.2
- cudnn 8.1.0

Details on where to find and how to install the the dependencies can be found here:

- [Tensorflow GPU guide](https://www.tensorflow.org/install/gpu)

- [cudnn install guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)


### Acquiring the data 
The project uses 3 main sets of data:
- Flights data, provided by [EUROCONTROL](https://www.eurocontrol.int/dashboard/rnd-data-archive) -  (For graders of the capstone project this is provided in the readme.txt)
- Weather data provided by [NCEI](https://www.ncei.noaa.gov/) - retrieved by the programme automatically.
- Airport information (Coordinates etc) - provided in the repository for the Europe's top 50 airports.

The usage of each of these is summarised in the chart below. Many of the functions in extract have a 'generate/write/read' capability meaning they generate the full data on the first cold run and return the stored filtered data from the data on subsequent runs. This is indicated by the hollow arrows in the chart below. Some of the functions can take long to generate, for example generating the weather data and Neural Network data for the top 50 airports can take up to an hour each. For capstone graders, reduced versions of the the filtered datasets are provided in the readme.txt. For potential legal reasons they are not provided publically in this repository.

#### Extraction chart
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

The project features 4 main end-user functions to process raw EUROCONTROL data (See [**the extraction chart**](#extraction-chart)):
- linearRegressionFormat() - filters flight data to only flights relevant to the [**Individual flight prediction**](#individual-flight-prediction). This is featured in the Randomforest jupyter notebook.
- generateNNdata() and a multi-airport wrapper generateNNdataMultiple() - aggregates flight data into timeslots and generates some engineered features. This is used in [**Single airport prediction**](#single-airport-prediction) and [**Graph Neural Network**](#graph-neural-network). The ExtractNN jupyter notebook showcases the use of these functions
- getAdjacencyMatrix() and distance_weight_adjacency() - generate different forms of adjacency matrices used in [**Graph Neural Network**](#graph-neural-network).

## Models
### Individual flight prediction
A Random Forest regression model was used to obtain delays at individual airports. Features such as airline, planned arrival time and airport capacity were used as input to predict the target variable, which is *arrival delay*. 

To obtain the Random Forest model, you can run the notebook RandomForest.ipynb. During the first run, two arrays will be generated containing the features and labels. If desired, these arrays can be saved in a csv file by specifying ``` save_to_csv = True ```. Then, in the next cell the first line can be uncommented to load in these arrays. By running the subsequent cells, the model will be tuned and its accuracy will be provided, along with several plots.

### Single airport prediction

In order to access the code for the single airport prediction (for incoming aircraft's arrival delays and for departing aircraft's departure delays), the user should access the file named LSTM_model.ipynb. This file consists of a Jupyter notebook sontaining cells for the separate parts of the code, such as generating the data, formatting the data, creating the model etc. For each cell, there are accompanying explanations which are meant to provide the user with the necessary information for understanding how the code is organised and how it works. 

### Graph Neural Network

TODO EDIT THIS SECTION

## Troubleshooting and Contact
TODO MAYBE OR MAYBE NOT ADD THIS
