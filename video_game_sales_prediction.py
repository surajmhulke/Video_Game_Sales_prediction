# -*- coding: utf-8 -*-
"""Video_Game_Sales_prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tc9I7bxLJWCAEnqyPVi1Y2nIRLz3hNxR

# **Overview**.    

The gaming industry is certainly one of the thriving industries of the modern age and one of those that are most influenced by the advancement in technology. With the availability of technologies like AR/VR in consumer products like gaming consoles and even smartphones, the gaming sector shows great potential. In this hackathon, you as a data scientist must use your analytical skills to predict the sales of video games depending on given factors. Given are **8 distinguishing factors** that can influence the sales of a video game.

**Data Description**:-


Train.csv –  3506 observations.     
Test.csv –  1503 observations.    
Sample Submission – Sample format for the submission.    
**Target Variable**: SalesInMillions
"""

#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')

# Read the csv files
input = pd.read_csv("/content/Train.csv")

#print all columns to understand the dataset
input.head()

"""# Data cleaning"""

input.isnull().sum()

"""There are no null values in the dataset. So we can move to the next step of removing unnecessary columns.

From dataset, we can observe that except `id` column, all the other columns play a significant role in final sales of videogames. So it can be dropped.
"""

input = input.drop(columns=['ID'])
train, test = train_test_split(input, test_size=0.2, random_state=42, shuffle=True)

"""# Descriptive Statistics"""

train.shape, test.shape

train.nunique()

#If you are seeing the output below for the first time visit this link
#to understand what the values in each of this rows(mean, std, min, max) actually

train.describe()

"""From above table, my first insight is I can create bar charts of **console, year**, **category** and **ratings** columns easily. For other columns I might have to go for some other visual representation since the the number of unique values is high.

# EDA

I am first opting for auto EDA packages like pandas-profiling for generating visualisations and there corresponding reports.
"""

!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

from pandas_profiling import ProfileReport
report = ProfileReport(train, title="Report", html={'style': {'full_width':True}}, explorative=True, missing_diagrams={'bar': True})

report.to_notebook_iframe()

#Save the report in file
report.to_file("pandas_profiling_report.html")

"""From the above reports we can gain following insights:-   
*   Console column graph:   
<img src="https://res.cloudinary.com/dk22rcdch/image/upload/v1595439244/VideoGameDatasetAnalysisImages/Screenshot_2020-07-22_at_11.02.44_PM_nxz5cm.png" width=400>      
The sales of **PS2** were the highest in the data set

*   Years Column graph:   
<img src="https://res.cloudinary.com/dk22rcdch/image/upload/v1595439371/VideoGameDatasetAnalysisImages/Screenshot_2020-07-22_at_11.05.51_PM_ycn3nl.png" width=400>  
The sales were highest between the period **2005-2010**.

*   Game category column graph:   
<img src="https://res.cloudinary.com/dk22rcdch/image/upload/v1595439531/VideoGameDatasetAnalysisImages/Screenshot_2020-07-22_at_11.08.40_PM_ugwpdi.png" width=400>   
  **Action** category games are most popular

Now let's compare individual columns with target(SalesInMillions) column to gain a few more insights into the data.
"""

#Sales of games that happened corresponding to each console.
df = pd.DataFrame(train.groupby(['CONSOLE']).agg({'SalesInMillions': 'sum'}))

df.plot.bar(figsize=(12, 6))

"""**💡Insight**:  From the above graph we can see that sales were highest for PS3 platform followed by Xbox360"""

df = pd.DataFrame(train.groupby(['YEAR']).agg({'SalesInMillions': 'sum'}))

df.plot.bar(figsize=(12, 6))

"""**💡Insight**:  From the above graph we can see that sales were highest in the year 2010"""

df = pd.DataFrame(train.groupby(['CATEGORY']).agg({'SalesInMillions': 'sum'}))

df.plot.bar(figsize=(12, 6))

"""**💡Insight**:  From the above graph we can see that sales were highest for action genre

# Model training
"""

!pip install catboost

import catboost as cat
cat_feat = ['CONSOLE','CATEGORY', 'PUBLISHER', 'RATING']
features = list(set(train.columns)-set(['SalesInMillions']))
target = 'SalesInMillions'
model = cat.CatBoostRegressor(random_state=100,cat_features=cat_feat,verbose=0)
model.fit(train[features],train[target])

"""# Model Accuracy"""

y_true= pd.DataFrame(data=test[target], columns=['SalesInMillions'])
test_temp = test.drop(columns=[target])

y_pred = model.predict(test_temp[features])

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_true, y_pred))
print(rmse)

import pickle
filename = 'finalized_model.sav'

pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

test_temp[features].head(1)

loaded_model.predict(test_temp[features].head(1))

