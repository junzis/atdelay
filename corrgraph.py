from extraction.extract import *

import pandas as pd
### 1 airport
airport = "EGLL"
airports50 = ICAOTOP50 # imported from extraction

# catagorical flightlength returns length in bins instead of an average
X = generateNNdata(airport, timeslotLength=15, catagoricalFlightDuration=False)
print(X.dtypes)
X = (X.drop("departuresArrivalDelay", axis=1)
    .rename(columns={"arrivalsFlightDuration": "arrFlDuration", "arrivalsDepartureDelay":"arrDepDelay", "arrivalsArrivalDelay":"arrArrDelay", "departuresFlightDuration":"arrFlDuration", "departuresDepartureDelay":"depDepDelay"}))
show_heatmap(X, dtkey="timeslot")
plt.show()