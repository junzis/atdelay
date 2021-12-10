from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split

sys.path.append(".")
from tools.tool_box import filtering_data_ordinal_enc
from tools.tool_box import parameter_search, get_data


X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)


def parameter_search():
    parameters = {"n_estimators": [150]}

    regr = RandomForestRegressor(n_jobs=-1)

    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    grid_search = GridSearchCV(
        regr, parameters, cv=cv, n_jobs=-1, verbose=4, scoring="neg_mean_absolute_error"
    ).fit(X_train, y_train)

    return grid_search.cv_results_


result = parameter_search()
print(result)
