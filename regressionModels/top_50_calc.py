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
        "EGBB": 40,
        "UKBB": 46,
        "LEAL": 36,
        "LPPR": 24,
        "LFMN": 50,
        "ESSA": 84,
        "LIME": 26,
        "LFLL": 48,
        "EIDW": 48,
        "EDDK": 36,
        "EDDS": 48,
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



def accuracy_calc(acc_dict: dict):
    return sum(acc_dict.values()) / len(acc_dict)


ICAOTOP50 = [
    "EGLL",
    "LFPG",
    "EHAM",
    "EDDF",
    "LEMD",
    "LEBL",
    "LTFM",
    "UUEE",
    "EDDM",
    "EGKK",
    "LIRF",
    "EIDW",
    "LFPO",
    "LOWW",
    "LSZH",
    "LPPT",
    "EKCH",
    "LEPA",
    "EGCC",
    "LIMC",
    "ENGM",
    "UUDD",
    "EGSS",
    "EBBR",
    "ESSA",
    "LGAV",
    "EDDL",
    "EDDT",
    "UUWW",
    "EFHK",
    "LEMG",
    "ULLI",
    "EPWA",
    "EGGW",
    "LSGG",
    "LKPR",
    "EDDH",
    "LHBP",
    "LTBA",
    "UKBB",
    "LEAL",
    "EGPH",
    "LROP",
    "LFMN",
    "LIME",
    "LPPR",
    "EDDS",
    "EGBB",
    "EDDK",
    "LFLL",
]
new_dict = {
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
    "EGLL": 5.889489890910065,
    "LFPG": 3.795442331570614,
    "EHAM": 4.483325306858709,
    "EDDF": 4.022649400353656,
    "LEMD": 5.261546667909832,
    "LEBL": 4.401206127258698,
    "LTFM": 5.198203077609934,
    "EDDM": 3.856881491942465,
    "EGKK": 5.682978532681429,
    "LIRF": 4.309078262385989,
    "ULLI": 3.6039484816696996,
    "UUEE": 7.8169804906927665,
    "UUWW": 2.7243991225837423,
    "LROP": 4.4270708591669266,
    "UUDD": 4.244190217083899,
    "EDDT": 3.6870329695533064,
    "EGPH": 4.153829718562654,
    "EGBB": 4.456718787221311,
    "UKBB": 4.422615503022433,
    "LEAL": 4.225588940008024,
    "LPPR": 4.60775865949576,
    "LFMN": 4.574162220773571,
    "ESSA": 4.2143734708999,
    "LIME": 4.432959538778985,
    "LFLL": 4.040411700926016,
    "EIDW": 5.696720079868284,
    "EDDK": 3.8098529923643456,
    "EDDS": 3.591418887028652,
}