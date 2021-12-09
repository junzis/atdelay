from extract import *


a = readCSV("filteredData", "general.csv")

# b = a.groupby(['ADEP','ADES']).size().reset_index().rename(columns={0:'count'})
# saveToCSV(b, "airportcombinations", "filteredData")
# print(len(b))
# print(b)

# c = (
#     a.drop_duplicates(["ADEP"])
#     .loc[:, ["ADEP", "ADEPLat", "ADEPLong"]]
#     .rename(columns = {"ADEP": "code", "ADEPLat": "longitude", "ADEPLong": "latitude"})
#     .sort_values(["code"])
#     .reset_index(drop=True)
# )
# saveToCSV(c, "airportlocations", "filteredData")
# print(c)


# d = (
#     a.drop_duplicates(["ADEP", "ADES"])
#     .loc[:, ["ADEP", "ADEPLat", "ADEPLong", "ADES", "ADESLat", "ADESLong"]]
#     # .rename(columns = {"ADEP": "code", "ADEPLat": "longitude", "ADEPLong": "latitude"})
#     # .sort_values(["code"])
#     .reset_index(drop=True)
# )

d = (
    a.groupby(['ADEP','ADES'], as_index=False)
    .agg({'ADEP': ['count', 'first'], "ADES":"first", "ADEPLat": "first", "ADEPLong": "first", "ADESLat": "first", "ADESLong": "first"})  
)
d.columns = ["count", "ADEP", "ADES", "ADESLong", "ADEPLong", "ADESLat", "ADEPLat"]
d = d.query("ADEP != ADES")
# d.columns.droplevel(0)
print(d.columns)
saveToCSV(d, "airportcombinations", "filteredData")
print(len(d))
print(d)