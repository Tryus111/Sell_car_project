#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:39:00 2023

@author: user
"""

import pandas as pd

selling_car_data = pd.read_csv('train.csv')
df_car = selling_car_data.copy()
df_car = df_car.dropna()

# Clean Unknown Data
df_car = df_car[df_car['engine_size'] != 0]
df_car = df_car[df_car['fuel_type'] != 'Unknown']
df_car = df_car[df_car['drivetrain'] !='Unknown']

# Drivetrain
df_car['drive_train_dummies'] = df_car['drivetrain'].apply(lambda x: 1 if x == 'Four-wheel Drive' else 0)

# Engine Size 
# High Performance > 2.5
# Low - Mid Performance < 2.5
df_car['performance engine'] = df_car['engine_size'].apply(lambda x: 1 if x > 2.5 else 0)

# Fuel Type
df_car['fuel_type'].value_counts()
def map_fuel_type(x):
    if x == 'Gasoline':
        return 1
    elif x == 'Hybrid':
        return 2
    elif x == 'E85 Flex Fuel':
        return 3
    elif x == 'Diesel':
        return 4
    elif x == 'Electric':
        return 5
    elif x == 'Flex Fuel':
        return 6
    else:
        return None
df_car['fuel_type_dummies'] = df_car['fuel_type'].apply(map_fuel_type)

# MPG
# Miles Per Gallon < 20 = below average
# Miles 20 < Per Gallon < 45 = average
# Miles Per Gallon > 45 = High
def avg_mpg(df_car):
    min_mpg = df_car['min_mpg'] 
    max_mpg = df_car['max_mpg']
    average_mpg = (min_mpg + max_mpg) / 2
    if average_mpg <= 20:
        return 0
    elif 20 < average_mpg < 45:
        return 1
    elif average_mpg > 45:
        return 2
    else:
        return None
    
df_car['efficiency'] = df_car.apply(avg_mpg, axis = 1)
df_car['efficiency'].value_counts()

# Save to csv 
df_car.to_csv('data_cleaned.csv', index = False)

