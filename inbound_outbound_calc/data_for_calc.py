import sys

sys.path.append(".")
from tools.tool_box import data_filter_ADEPADES
from datetime import datetime
import pandas as pd
import numpy as np


df = pd.read_csv("LRData/LRData.csv", header=0, index_col=0)
print(df.head)
df_2 = data_filter_ADEPADES(df, datetime(2019, 3, 1), datetime(2019, 3, 2), "EGLL")

df_2.to_csv("cap_cal_data.csv")
