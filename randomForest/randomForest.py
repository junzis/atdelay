from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def parameter_search(n_estimators: int = 1, min_samples_leaf: int = 10):
    parameters = {"n_estimators": [5, 10, 25, 50, 100, 150, 250, 500]}

    regr = RandomForestRegressor(n_jobs=-1)
    
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    grid_search = GridSearchCV(
        regr, parameters, cv=cv, n_jobs=-1, verbose=4, scoring="neg_mean_squared_error"
    ).fit(X_train, y_train)