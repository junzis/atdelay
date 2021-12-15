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


def top50_calculator():
    airport_dict = {
        "LEAL": 36,
        "EDDS": 48,
        "LFLL": 48,
        "ESSA": 84,
        "EGPH": 42,
        "UKBB": 36,
        "EIDW": 48,
        "LIME": 26,
        "EDDK": 36,
        "LFMN": 50,
        "EGBB": 40,
        "LPPR": 24,
    }
    acc_dict = {}
    for airport in airport_dict:
        print(f"Doing calculations for aiport: {airport}")
        X, y = filtering_data_onehot(
            filename="LRData/LRDATA.csv",
            start=datetime(2018, 1, 1),
            end=datetime(2019, 12, 31),
            airport=airport,
            airport_capacity=airport_dict[airport],
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
    return acc_dict


acc_dict = {
    "LGAV": 4.472835467211704,
    "EDDL": 4.084764518650214,
    "EFHK": 3.95690555508362,
    "LEMG": 4.7155325377479285,
    "EPWA": 4.519428133965428,
    "EGGW": 4.583746230814193,
    "LSGG": 4.4874583044872365,
    "LKPR": 4.170584482540404,
    "EDDH": 3.7267474546350816,
    "LHBP": 4.107814833219582,
    "LTBA": 7.156262372123445,
    "EGLL": 5.890174064574497,
    "LFPG": 3.795944781190017,
    "EHAM": 4.491391663927601,
    "EDDF": 4.021868158765804,
    "LEMD": 5.264608351459311,
    "LEBL": 4.397343530657209,
    "LTFM": 5.192742405806272,
    "EDDM": 3.8597441115803255,
    "EGKK": 5.680251456099521,
    "LIRF": 4.311629059242113,
    "LFPO": 4.052652823141702,
    "LOWW": 4.443855332164622,
    "LSZH": 4.711077381748813,
    "LPPT": 5.54990567442422,
    "EKCH": 3.9218357147589145,
    "LEPA": 4.279129192931119,
    "EGCC": 4.3926117886460005,
    "LIMC": 4.579961412211301,
    "ENGM": 4.107187693201358,
    "EGSS": 4.314323616406326,
    "EBBR": 4.0016467162512095,
    "LGAV": 4.4713597569988295,
    "EDDL": 4.090057688354618,
    "EFHK": 3.9463324275420084,
    "LEMG": 4.708581142680042,
    "EPWA": 4.514849847150704,
    "EGGW": 4.59130711414351,
    "LSGG": 4.491527543092178,
    "LKPR": 4.1656676025023724,
    "EDDH": 3.722949885869376,
    "LHBP": 4.110922207032962,
    "LTBA": 7.14779456112664,
    "ULLI": 3.617802239056201,
    "UUEE": 7.817461894330097,
    "UUWW": 2.71295466056362,
    "LROP": 4.432418613465295,
    "UUDD": 4.225282732445331,
    "EDDT": 3.6809168239349703,
    "LEAL": 4.227783692989105,
    "EDDS": 3.594763551309093,
    "LFLL": 4.032988388793106,
    "ESSA": 4.218605246704379,
    "EGPH": 4.158158841857587,
    "UKBB": 4.4298015541303215,
    "EIDW": 5.682570745407447,
    "LIME": 4.446231944425944,
    "EDDK": 3.804770333402046,
    "LFMN": 4.571354555936264,
    "EGBB": 4.468759992134821,
    "LPPR": 4.597604797100962,
}


def accuracy_calc(acc_dict: dict):
    return sum(acc_dict.values()) / len(acc_dict)


print(accuracy_calc(acc_dict))
# print(set(ICAOTOP50).difference(set(acc_dict)))
