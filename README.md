# Video_Game_Sales_prediction
# Overview.
The gaming industry is certainly one of the thriving industries of the modern age and one of those that are most influenced by the advancement in technology. With the availability of technologies like AR/VR in consumer products like gaming consoles and even smartphones, the gaming sector shows great potential. In this hackathon, you as a data scientist must use your analytical skills to predict the sales of video games depending on given factors. Given are 8 distinguishing factors that can influence the sales of a video game.

Data Description:-

Train.csv – 3506 observations.
Test.csv – 1503 observations.
Sample Submission – Sample format for the submission.
Target Variable: SalesInMillions


# Step 1: Identifying target and independent features
First, let’s import Train.csv into a pandas dataframe and run df.head() to see the columns in the dataset.
Sales Prediction Web Application : Dataset

Column values
From the dataframe, we can see that the target column is SalesInMillions and rest of the columns are independent features

# Step 2: Cleaning the data set
First, we check for null values by running input.isnull().sum() command.
input.isnull().sum()
#Output:
ID                 0
CONSOLE            0
YEAR               0
CATEGORY           0
PUBLISHER          0
RATING             0
CRITICS_POINTS     0
USER_POINTS        0
SalesInMillions    0
dtype: int64
We can see that there are no null values in the dataset. Next, we can drop unnecessary ID column since it does not play a role in target sales by running below command:- input = input.drop(columns=['ID'])
Next, we can split the dataframe into training and test dataset using train_test_split command:-
train, test = train_test_split(input, test_size=0.2, random_state=42, shuffle=True)

# Step 3: Exploratory Data Analysis
Descriptive Statistics
Using df.shape command we can find a count of total rows in the dataset and df.nunique() command can be used to find unique values in each of the columns.
CONSOLE              17
YEAR                 23
CATEGORY             12
PUBLISHER           184
RATING                6
CRITICS_POINTS     1499
USER_POINTS        1877
SalesInMillions    2804
In the EDA section, we make use of pandas profiling and matplotlib packages to generate graphs of various columns and observe their relationships with the target column.
A few insight gained from EDA are:-
Sales were highest for the PS3 platform. It was followed by Xbox360:
Sales Prediction Web Application : Sales
Sales were highest for the action category and lowest for puzzles
Sales
And sales were highest in the year in the period from 2007 to 2011:
Sales Prediction Web Application : Sales
Usually, we go for feature engineering or feature selection steps after EDA. But we have fewer features and emphasis on actually using the model. So we are moving forward towards the next steps. However, keep in mind that USER_POINTS and CRITICS_POINTS columns can be used to derive extra features.

# Step 4: Building a model
We are going to use catboost regression model for our dataset since we have a lot of categorical features. This skips the step of label encoding categorical features since catboost can work on categorical features directly.
First, we install catboost package using pip install command.
Then we create a list of categorical features, pass it over to the model and then fit the model on train dataset:
import catboost as cat
cat_feat = ['CONSOLE','CATEGORY', 'PUBLISHER', 'RATING']
features = list(set(train.columns)-set(['SalesInMillions']))
target = 'SalesInMillions'
model = cat.CatBoostRegressor(random_state=100,cat_features=cat_feat,verbose=0)
model.fit(train[features],train[target])

# Step 5: Check model accuracy
First, we create true predictions from test dataset:
y_true= pd.DataFrame(data=test[target], columns=['SalesInMillions'])
test_temp = test.drop(columns=[target])
Next, we run our trained model on test dataset to get model predictions and check model accuracy
y_pred = model.predict(test_temp[features])
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_true, y_pred))
print(rmse)
#Output: 1.5555409360901584
We have an RMSE value of 1.5 which is pretty decent. For more information about accuracy metrics in case of regression problems, you can refer to this article. If you would like to improve the model further or try to combine various models you can refer to the approaches of the winners of this hackathon in this article: Analytics Vidya
 

# Step 6: Save the model into a pickle file
We can now save our model into a pickle file and then save it locally:
import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model.predict(test_temp[features].head(1))
array([2.97171105])
