# Stock Price Prediction with LSTM

This repository contains a stock price prediction model implemented in Python using Long Short-Term Memory (LSTM) neural networks. The model predicts the closing prices of a given stock based on historical data.

## Introduction

Stock price prediction is a common application of machine learning and deep learning in finance. LSTM networks, a type of recurrent neural network (RNN), are well-suited for sequence prediction tasks such as time series forecasting.

## Dataset

The dataset used in this project is historical stock price data obtained from the `tesla.csv` file. It contains the following columns:
- Date: The date of the stock price data
- Close: The closing price of the stock on that date

## Installation

To run the code in this repository, you need to have Python installed along with the following libraries:
- numpy
- pandas
- matplotlib
- pandas_datareader
- scikit-learn
- keras

You can install these dependencies using pip:
pip install numpy pandas matplotlib pandas_datareader scikit-learn keras


## Usage

1. Clone this repository:
git clone https://github.com/shreyasaini003/stock_prediction


2. Navigate to the cloned directory:
cd stock_prediction


3. Run the script `stock_prediction.py`:


## Model Architecture

The LSTM model architecture used in this project consists of the following layers:
- LSTM layer with 64 units and return sequences enabled
- LSTM layer with 64 units and return sequences disabled
- Dense layer with 32 units
- Output layer with 1 unit

## Results

After training the model, it is evaluated on the test set to measure its performance. The predictions are then compared with the actual stock prices to visualize the accuracy of the model.

## Example

An example of predicting stock prices and visualizing the results is provided in the script. You can modify the code to use different stocks or adjust parameters to experiment with the model.



