import pandas as pd
import matplotlib.pyplot as plt

def convertTime(input):
    mins = float(input.split("m")[0])
    seconds = float(input.split("m")[1][:-1])
    result =  mins + seconds/60
    return result

time = ["0m0s", "2m31.435s", "3m44.822s", "5m15.252s", "5m34.728s", "7m44.655s", "7m44.600s", "9m54.563s", "8m44.928s", "13m15.318s", "12m9.633s"]
days = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

time_converted = list(map(convertTime, time))
d = {"Time": time_converted, "Days": days}
df = pd.DataFrame(d)

# plt.rcParams.update({'font.size': 13})
plt.figure(figsize=(12, 8))
plt.plot("Days", "Time", data=d)
plt.xlim(-.2, 10.2)
plt.ylim(-.4, 14.4)
plt.xticks(days)
plt.xlabel("Amount of data by Day (each day ~ 1.6GB compressed)")
plt.ylabel("Process Time by Minute")
plt.title("Spark Processed Time by amount of data \n (Number of Executor = 10, Memory each executor = 2GB)",
          fontweight="bold")
plt.savefig("./AssignmentWeek4/BENCH-s2845016-s2800209-BENFSD.png")