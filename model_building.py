#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:28:14 2023

@author: user
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

df_eda = pd.read_csv('data_cleaned.csv')
df_eda.columns

# Relevant Columns
df_model = df_eda[['engine_size', 'automatic_transmission',
       'damaged', 'first_owner', 'personal_using',
       'turbo', 'alloy_wheels', 'adaptive_cruise_control', 'navigation_system',
       'power_liftgate', 'backup_camera', 'keyless_start', 'remote_start',
       'sunroof/moonroof', 'automatic_emergency_braking', 'stability_control',
       'leather_seats', 'memory_seat', 'third_row_seating',
       'apple_car_play/android_auto', 'bluetooth', 'usb_port', 'heated_seats',
       'price', 'drive_train_dummies',
       'performance engine', 'fuel_type_dummies', 'efficiency']]

df_select = ['engine_size', 'price', 'drive_train_dummies',
             'performance engine', 'fuel_type_dummies', 'efficiency' ]

# Mencari nilai yang hilang
missing_values = df_model.isna().sum()
print(missing_values)

# Menghapus baris dengan nilai hilang
df_model = df_model.dropna()

# Menggantikan nilai infinit dengan nilai yang valid
df_model = df_model.replace([np.inf, -np.inf], np.nan)
df_model = df_model.dropna()

# dummies
df_dum = pd.get_dummies(df_model)

x1 = df_model.drop('price', axis = 1)
y = df_model['price']

x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size = 0.1, random_state = 365)

# Linear Regression
X = sm.add_constant(x1)
model = sm.OLS(y, X)
results = model.fit()
results.summary()
