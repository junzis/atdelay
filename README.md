# Machine Learning Air Traffic Delays Prediction Models

This repository includes the source code of Machine learning models for predicting air traffic delays at different airport and the network of airports, using the following models:

1. LSTM model: Predicts aggregated arrival and departure delays for a single airport
2. DST-GAT model: Predicts aggregated arrival and departure delays of all airports a network


## Models


### Single airport prediction

Single airport delay prediction focuses on the arrival and departure delays of any given airport using a long-short term memory recurrent network (LSTM) model.

The code for this model can be found in the notebook: `lstm.ipynb`.

### Graph Neural Network

The dynamic spatial-temporal graph attention network (DST-GAT) is a model which is used to predict the delays for a network of airports. This is done based on the connection between airports and the history of every airport. For every airport, the departure and arrival delays are predicted for several lookahead time steps. 

The code for this model can be found in the notebook: `dstgat.ipynb`.


## Python library dependencies

The following libraries are needed to run the code in this repository:

- numpy
- scipy
- pandas
- scikit_learn
- keras
- spektral
- tensorflow


### Configuring TensorFlow GPU functionality (Optional)
Training neural networks and in particular, Graph Neural Networks tends to be quite slow. We therefore also recommend configuring TensorFlow to work with a graphics card (CUDA compatible NVIDIA GPU required). To do this some additional dependencies should be installed (see Tensorflow GPU guide). Importantly, all the versions should be compatible.

Details on where to find and how to install the dependencies can be found here:

- [Tensorflow GPU guide](https://www.tensorflow.org/install/gpu)
- [cudnn install guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)


## Data 
Currently, the tools use 2 main sets of data:
- [EUROCONTROL R&D data](https://www.eurocontrol.int/dashboard/rnd-data-archive): not provided with this software. You need to request a license and download the data on your own.
- Airport coordinates: provided in `tools.constants.py` for European top 50 airports.

Many of the functions in `tools/extract.py` have generate, write, and read capabilities, which means they generate the full data on the first cold run and return the stored filtered data from the data on subsequent runs.


Your file tree should look something like this:
```
[repostiont root]
├───data
│   ├───2018
│   │   ├───201803
│   │   ├───201806
│   │   ├───201809
│   │   └───201812
│   ├───2019
│   │   ├───201903
│   │   ├───201906
│   │   ├───201909
│   │   └───201912
│   ...
|
├───[other files]
.   .
.
.
.
```


## Contributors

This software was originated from a TU Delft's Capstone Project (TI3150TU). The following people contributed to the orignal code:

* Junzi Sun [@junzis](https://github.com/junzis) (project tutor)
* Constantinos Aristodemou [@ConstantinosAr](https://github.com/ConstantinosAr)
* Vlad Buzetelu [@vladbuzetelu](https://github.com/vladbuzetelu)
* Tristan Dijkstra [@IrTrez](https://github.com/IrTrez)
* Theodor Falat [@theofalat](https://github.com/theofalat)
* Tim Hogenelst [@TimGioHog](https://github.com/TimGioHog)
* Niels Prins [@Niels](https://github.com/Niels)-Prins 
* Benjamin Slijper [@BenjaminSlijper](https://github.com/BenjaminSlijper)

