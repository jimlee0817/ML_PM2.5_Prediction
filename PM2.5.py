#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 23:08:32 2021

@author: jimlee
"""


import csv
import numpy as np

# open the csv file
rawData = np.genfromtxt('train.csv', delimiter=',')
data = rawData[1:,3:] # data is ready, but need to be reorganized




    