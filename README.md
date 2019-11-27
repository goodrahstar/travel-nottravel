# travel audience Data Science Challenge

## Goal

One of the main problems we face at travel audience is identifying users that will eventually book a trip to an advertised destination. In this challenge, you are tasked to build a classifier to predict the conversion likelihood of a user based on previous search events, with emphasis on the feature engineering and evaluation part.


## Solution
I have selected following three features to build the model:

* `days_before_plan` : The factor when user is looking for a flight plays and important role in deciding if they will book or not. Mostly if the days are very far then that represents user was just checking for price or was performing other investigations.
* `origin_enc` & `destination_enc`: the location also plays an important role since some of the destination are quite popular then others.
* `trip_duration`: How long user is the trip also plays important role when combined with other secondary features like `distance`, `num_adults`, `num_children`.


To build the model i have used Neural Network architecture with 1 hidden layer, trained over `Adam` optimizer. I have also used early stop mechanism so that we dont perform unnessasry computation if model is not converging.

## Processed Data 

Provided raw data was transformed into featues with followong columns:

- `Finaldata_after_encoding.csv` - The processed data is stored in this file with following features. This processed data is directly used into the model training process.
  * `num_adults` - number of adults
  * `num_children` - number of children
  * `days_before_plan` - number of days before the user is planning to book the flight
  * `trip_duration` - number of days the trip will last long
  * `distance` -  distance between the origin and destination
  * `origin_enc` - IATA airport code of the origin airport is encoded into numeric form
  * `destination_enc	` - IATA airport code of the destination airport is encoded into numeric form
  * `event_type_enc` - Label data `search` / `book` encoded into numeric form

## Steps for execution
Install dependencies in Py3.7 env
 `pip install -r requirements.txt`

Execute testing
 `python tester.py`

Tensorboard Log: https://tensorboard.dev/experiment/ieKEqL2rQ3mD9vYFWCAF7w

Training Log in tensorboard
![Screenshot](https://github.com/goodrahstar/travel-nottravel/raw/master/data/Screenshot%202019-11-27%20at%2010.19.29%20AM.png)
