#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 12:53:24 2019

@author: rahulkumar


## Tasks

Your code needs to do the following:

- data preparation:
  - calculate the geographic distance between origins and destinations
  - convert raw data to a format suitable for the classification task
- feature_engineering:
  - based on the given input data, compute and justify three features of your choice that are relevant for predicting converters
- experimental design:
  - split data into test and training sets in a meaningful way
- model:
  - a classifier of your choice that predicts the conversion-likelihood of a user

Use your best judgment to define rules and logic to compute each feature. Don't forget to comment your code!


"""

import pandas as pd
import math
from datetime import datetime


 

#Keras Imports
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



class CustomerConversion():
    def __init__(self, event_file, iata_file):
        self.event_file = event_file
        self.iata_file = iata_file
        self.process_iata()
        
        
        
    def process_iata(self):
        # Processing the iata file and converting it into a dict format 
        self.iata = pd.read_csv(self.iata_file)
        self.iata.set_index("iata_code", inplace = True)
        self.iata = self.iata.to_dict()
    
    
    def calc_distance(self,origin, destination):
        # This method is responsible to calculate the geographic distance 
        # between origins and destinations in KM

        lat1, lon1 = self.iata['lat'][origin] , self.iata['lon'][origin]

        lat2, lon2 = self.iata['lat'][destination] , self.iata['lon'][destination]
        
        radius = 6371 # km
        # Haversine formula
        dlat = math.radians(lat2-lat1)
        dlon = math.radians(lon2-lon1)
        a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = radius * c

        return d
    
    
    def calc_no_of_days(self,date_from,looking_on):
        # This method is responsible to calculate the number of days between 2 dates.
        
        date_format = "%Y-%m-%d"
        a = datetime.strptime(looking_on, date_format)
        b = datetime.strptime(date_from, date_format)
        delta = b - a #days
        return delta.days
        
    def prepare_training_data(self):
        # Here we are processing the input data, the final output will be all the numerical representations
        
        event_data = pd.read_csv(self.event_file)
        event_data = event_data.dropna()

        # This feature 'look_on' will be used to calculate the no of days before the user is planning to book/search for a flight.
        event_data['look_on']=event_data['ts'].str.split(" ", n = 1, expand = True)[0]        
        event_data['days_before_plan'] = event_data.apply(lambda row : self.calc_no_of_days(row['date_from'], row['look_on']), axis = 1)
        
        # This feature 'trip_duration' represents for how many days the trip was planned for
        event_data['trip_duration'] = event_data.apply(lambda row : self.calc_no_of_days(row['date_to'], row['date_from']), axis = 1)

        # This feature is the distance between the  origin and destination. 
        # I dont think this can be a primary feature but still would like to have it as extra metadata for model to learn
        event_data['distance'] = event_data.apply(lambda row : self.calc_distance(row['origin'], row['destination']), axis = 1)
        
        
        self.event_data = event_data.drop(columns=['ts','user_id'])

        self.transform_training_data()

        
    def transform_training_data(self):
        # This method is responsible for transformaing all the 'string' objects and encode them into numerical representation
        
        # Encoding origin column
        le = LabelEncoder()
        le.fit(self.event_data['origin'])
        origin_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))
        self.event_data['origin_enc'] = le.transform(self.event_data['origin'])
        
        #Encoding destination column
        le = LabelEncoder()
        le.fit(self.event_data['destination'])
        destination_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))
        self.event_data['destination_enc'] = le.transform(self.event_data['destination'])
        
        # Encoding our label column i.e 'event_type'
        le = LabelEncoder()
        le.fit(self.event_data['event_type'])
        event_type_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))
        self.event_data['event_type_enc'] = le.transform(self.event_data['event_type'])
        temp_df= pd.DataFrame()
        temp_df['label'] = self.event_data['event_type_enc']
        # onehot transformation
        onehotencoder = OneHotEncoder()
        encoded_labels_data = onehotencoder.fit_transform(temp_df).toarray() 
        
        # Removing unwanted columns
        self.event_data = self.event_data.drop(columns=['date_from','date_to','origin','destination','look_on','event_type'])

        # Preparing input data for the model
        encoded_input = self.event_data.iloc[:, 0:-1].to_numpy()
        self.input_dim = encoded_input.shape[1]
        
        # Splitting the data into 80-20 ratio for training and testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( encoded_input, encoded_labels_data, test_size=0.2, random_state=42)
        

    def initialize_model(self):

        # NN Architecture    
        model = Sequential()
        model.add(Dense(7, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
        self.model = model
        

        
    def train(self):
      
        # Training parameter  
        epochs = 30  # TODO: Make autoconfigurable
        batch_size = 6
        
        # Preparation of Training Data from csv
        self.prepare_training_data()
        
    
        # Initialize model
        self.initialize_model()
        
        # Defining callbacks
        checkpoint = ModelCheckpoint('./model/weight.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./model')
        early_stop = EarlyStopping(monitor='val_accuracy' , patience=7)
        
        # Optimizer for our model
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
        
        
        # Training the model
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])    
        
        
        self.history = self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint,tensorboard_callback,early_stop], validation_data=(self.X_test, self.y_test))  # starts training
        
        return self.history
    
    