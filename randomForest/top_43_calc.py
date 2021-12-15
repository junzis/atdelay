from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
sys.path.append(".")
from datetime import datetime
from tools.tool_box import filtering_data_onehot

airport_dict = {}

acc_dict = {}
for airport in airport_dict:
    print(f"Doing calculations for aiport: {airport}")
    X, y = filtering_data_onehot(filename= "LRData/LRDATA.csv",
    start=datetime(2018, 1, 1),
    end= datetime(2019, 12, 31),
    airport =  airport,
    airport_capacity = airport_dict[airport]
)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    forest = RandomForestRegressor(
        n_estimators=300,
        max_features="auto",
        max_depth=50,
        min_samples_split=10,
        min_samples_leaf=2,
        bootstrap=True,
        n_jobs=-1,
    )
    forest.fit(X_train, y_train)
    prediction = forest.predict(X_test)
    score = mean_absolute_error(y_test, prediction)
    print(f"Accuracy of {airport} = {score}")
    acc_dict[airport] = score
    print(f"accuracy dictionary until now = {acc_dict}")

print(acc_dict)
