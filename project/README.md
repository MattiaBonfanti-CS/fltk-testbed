# CS4215-QPECS Project
CS4215 Quantitative Performance Evaluation for Computing Systems - Group Project

#### Adjusting Learning Rate and Epochs of Deep Neural Networks for Optimal Accuracy/Training Time Balance

When training a Deep Neural Network, the learning rate and the number of epochs are two of the most important parameters. 
The challenge rises when an optimal balance between accuracy of the results and training time has to be found. 
This work aims to show the impact of the learning rate, the epochs and the system configuration on the training accuracy and response time. 
A set of values for the learning rate, number of epochs, datasets and network type is used to design experiments that are then evaluated through a 2k and full factorial ANOVA analysis to establish the significance of the factors and their relations. 
Building on these results, several predictive models are then built to estimate the training accuracy, response time and best performing system configuration. 
The findings lead to an optimization strategy to find the best accuracy and response time balance, which leverages on the combination of variable epochs and arrival rates. 
The work adds an additional perspective to the literature on the topic of Deep Neural Networks and can serve as a starting point and support material for further studies.

## Structure 

```
| - anova: ANOVA analysis scripts
| - configs: Configuration files for experiments
| - connections: MongoDB connection scripts
| - models: Data structure classes
| - plots: Folder to store plots (TO BE CREATED)
| - queuing_theory: Queuing Theory analysis scripts
| - results: Experiments results
| - __init__.py: Initialization values
| - experiments_generator.py: Generate experiments configuration files
| - results_parser.py: Export the experiment results to MongDB
| - utils.py: Utility functions
```

## How To Run It

#### Requirements

| Name | Link |
|---|---|
| Python 3.9 | [Pyhton 3.9 Download and install](https://www.python.org/downloads/release/python-3915/) |
| PyCharm IDE (recommended) | [PyCharm installation](https://www.jetbrains.com/pycharm/download/) |
| MongoDB | [MongoDB Community installation](https://www.mongodb.com/try/download/community) |

#### Setup MongoDB

You can use either a local MongoDB or a cloud based one through Atlas. 
Make sure to include the following environment variables to be able to connect:

```
MONGODB_HOST
MONGODB_PORT
MONGODB_USER
MONGODB_PASSWORD
MONGODB_DATABASE
MONGODB_QPARAMS
```

Otherwise, you can create a `connections.cfg` file in the `connections` package with the following structure:

```
[mongodb]
HOST=...
PORT=...
USER=...
PASSWORD...
DATABASE=...
QUERY_PARAMS=...
```

The following collections will be created once the code runs (you can also create them manually):

- **doe_data**: Stores the experiments results.
- **anova_data**: Stores the ANOVA analysis results.
- **queue_data**: Stores the Queuing Theory estimations.

#### Clone

Clone this repo to your local machine using 
```shell script
git clone https://github.com/MattiaBonfanti-CS/fltk-testbed.git
cd fltk-testbed
git checkout cs4215_project
```

### Setup plots directory
Move to  the application folder, checkout the `cs4215_project` branch and run in your terminal:
```shell script
cd project
mkdir plots
```

#### Create Virtual Environment (venv)
Move to  the application folder, checkout the `cs4215_project` branch and run in your terminal:
```shell script
# Create virtualenv, make sure to use python3.9
$ virtualenv -p python3 venv
# Activate venv
$ source venv/bin/activate
```
Alternatively:
* Open the project with PyCharm (either Pro or CE)  or your favorite Python IDE
* Select python3.9 as project interpreter

#### Install Requirements
Move to  the application folder and run in your terminal:
```shell script
cd project
pip install -r requirements.txt
```

## ANOVA Analysis

Perform 2k-factorial, full-factorial and ANOVA linear predictive model to estimate accuracy.

```shell script
cd project
cd anova

# 2k-factorial
python 2_k_factorial.py

# full-factorial
python full_factorial.py

# linear model
python linear_model.py
```

## Queueing Theory

Estimate response time for queues of type: M/M/1; M/M/1-fast; M/M/k; M/M/k-fast. 
Compare performance of different queues and predict the response time using the linear regression model.

```shell script
cd project
cd queuing_theory

# estimate response time
python estimator.py

# compare performance
python comparison.py

# linear model
python linear_model.py
```

## Contributors

| Name | Email |
|---|---|
| Mattia Bonfanti | m.bonfanti@student.tudelft.nl |
| Ivan Todorov | i.todorov@student.tudelft.nl |
