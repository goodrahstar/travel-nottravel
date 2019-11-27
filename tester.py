#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 12:53:24 2019

@author: rahulkumar
 
"""
from model_keras_classification import CustomerConversion


cc = CustomerConversion(event_file='./data/events_4.csv', iata_file='./data/iata_2.csv')
                        
cc.train()


cc.model.predict(cc.X_test)
