import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from glob import glob

lookforwards = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

def importCSV(folderName="LSTM_Lookahead_Results", segments = lookforwards):
    listOfFiles = []
    listOfFiles.extend(glob(f"{folderName}/*"))
    filesDict = {}
    for idx, file in enumerate(listOfFiles):
        filesDict[segments[idx]] = pd.read_csv(file, header=0, skiprows=14, parse_dates=[0])#.to_numpy()

    return filesDict

P = importCSV()

def plotComparison(P, lookforward, windowSize=300):
    fig, axs = plt.subplots(2, 1, sharex=True, num=int(lookforward*10))
    # print(P.head(5))
    axs[0].plot(P.iloc[:windowSize, 0], P.iloc[:windowSize,2+1], label="Actual Arrival Delay")
    axs[1].plot(P.iloc[:windowSize, 0], P.iloc[:windowSize,0+1], label="Actual Departure Delay")
    axs[0].plot(P.iloc[:windowSize, 0], P.iloc[:windowSize,3+1], label="Predicted Arrival Delay")#, linestyle="--")
    axs[1].plot(P.iloc[:windowSize, 0], P.iloc[:windowSize,1+1], label="Predicted Departure Delay")#, linestyle="--")

    axs[0].legend()
    axs[1].legend()
    axs[1].set_xlabel("Time (hours)")
    axs[0].set_ylabel("Arrival Delay (mins)")
    axs[1].set_ylabel("Departure Delay (mins)")

    plt.suptitle(
        f"Comparison for: EGLL. Forward: {lookforward}h, Backward: {4}h"
    )
    axs[1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%H-%M"))
    plt.xticks(rotation=45)





for idx, u in enumerate(lookforwards):
    plotComparison(P[u], idx)

plt.show()
# def plotComparison(airport_index, hour):
#     # ypredFull, yactualFull, _, _, _, _, _, _, time = getLabelArrays(hour)
#     fig, axs = plt.subplots(2, 1, sharex=True, num=airport_index)


#     axs[0].plot(
#         time, yactualFull[:, 0 + airport_index, 0], label="Actual Arrival Delay"
#     )
#     axs[1].plot(time, yactualFull[:, airport_index, 1], label="Actual Departure Delay")
#     axs[0].plot(
#         time, ypredFull[:, 0 + airport_index, 0], label="Predicted Arrival Delay"
#     )
#     axs[1].plot(time, ypredFull[:, airport_index, 1], label="Predicted Departure Delay")
#     axs[0].legend()
#     # axs[1].legend()
#     axs[1].set_xlabel("Time (hours)")
#     axs[0].set_ylabel("Arrival Delay (mins)")
#     axs[1].set_ylabel("Departure Delay (mins)")
#     plt.suptitle(
#         f"Comparison for: {airports[airport_index]}. Forward: {hour}h, Backward: {input_sequence_length}h"
#     )
#     axs[1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
#     axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%H-%M"))
#     plt.xticks(rotation=45)


