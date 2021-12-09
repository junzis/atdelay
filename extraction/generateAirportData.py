from extract import *


a = readCSV("filteredData", "general.csv")

b = a.groupby(['ADEP','ADES']).size().reset_index().rename(columns={0:'count'})
saveToCSV(b, "airportcombinations", "filteredData")
print(len(b))
print(b)

c = (
    a.drop_duplicates(["ADEP"])
    .loc[:, ["ADEP", "ADEPLat", "ADEPLong"]]
    .rename(columns = {"ADEP": "code", "ADEPLat": "longitude", "ADEPLong": "latitude"})
    .sort_values(["code"])
    .reset_index(drop=True)
)
saveToCSV(c, "airportlocations", "filteredData")
print(c)