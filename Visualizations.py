from extraction.extract import *

import pandas as pd

### 1 airport
airport = "EHAM"
airports50 = ICAOTOP50 # imported from extraction

# catagorical flightlength returns length in bins instead of an average
X = generateNNdata(airport, timeslotLength=15, catagoricalFlightDuration=False)


show_heatmap(X, dtkey="timeslot")


show_raw_visualization(X,date_time_key="timeslot")

plt.show()