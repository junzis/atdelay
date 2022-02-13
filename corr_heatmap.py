from preprocess.extract import *
import seaborn as sns


airport = "EHAM"

X = generateNNdata(
    airport,
    start=datetime(2019, 1, 1),
    timeinterval=15,
    catagoricalFlightDuration=False,
)

DropThese = [
    "timeslot",
    "arrivalsDepartureDelay",
    "departuresArrivalDelay",
    "date",
]

ticks = ['Departing Flights', 'Arriving flights', 'Fraction low-cost', 'Mean arrival flight duration',     
       'Average arrival delay', 'Mean departure flight duration',
       'Average departure delay', 'Planes at airport', 'Filled capacity airport', 'Day of week',
       'Month', 'Hour of day']

X = X.drop(DropThese, axis=1).corr().round(2)
print(X.columns)

plt.figure(figsize=(9, 9))
plt.tight_layout()
heat_map = sns.heatmap(
    X,
    linewidth=1,
    annot=True,
    center=0,
    cmap="Spectral_r",
    vmin=-1,
    vmax=1,
    square=True,
    # xticklabels=ticks,
    yticklabels=ticks,
)
heat_map.set_xticklabels(ticks,rotation = 30, ha="right")
plt.show()
