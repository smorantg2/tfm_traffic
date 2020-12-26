import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

with open('vehicles_show.json') as f:
  data = json.load(f)

df = pd.DataFrame.from_dict(data["vehicles"])

col1 = []
for i in range(len(df)):
    col1.append(df.iloc[i,0][1])

df["vehicle"] = col1
df["time"] = df["time"]/60
df["speed"] = np.random.normal(loc=104.34, scale=8.12, size=len(df))

a = np.array(df["time"]).astype("int8")
hours = np.array([i for i in range(14)])
amount = [np.sum(a==i) for i in range(14)]


fig, ax = plt.subplots()
data = df["speed"]

N, bins, patches = ax.hist(data, edgecolor='black', linewidth=1,weights=np.ones(len(data)) / len(data))


patches[-1].set_facecolor('red')

plt.title("Velocidad media de los vehículos")
plt.xlabel("km/h")
plt.ylabel("Frecuencia")
plt.xlim(60,130)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
#plt.savefig("/home/srg/PycharmProjects/images/speed_hist.png")
plt.show()
plt.close()

fig, ax = plt.subplots()
data = df["vehicle"]

N, bins, patches = ax.hist(data, edgecolor='black', linewidth=1,weights=np.ones(len(data)) / len(data))
patches[0].set_facecolor('red')
patches[1].set_facecolor('blue')
patches[2].set_facecolor('purple')
patches[3].set_facecolor('green')
plt.title("Frecuencia por tipo de vehículo")
plt.ylim(0,1)
plt.xlabel("Tipo de vehículo")
plt.ylabel("Frecuencia")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
#plt.savefig("/home/srg/PycharmProjects/images/vehicle_hist.png")
plt.show()
plt.close()

plt.plot(hours, amount)
plt.fill_between(hours, amount, color="skyblue")
plt.plot(hours, amount, color="Slateblue", alpha=0.6)
plt.scatter(hours, amount, color="Slateblue", s = 5)
plt.title("Número de vehículos por horas")
plt.ylim(0,120)
#plt.savefig("/home/srg/PycharmProjects/images/vehicle_freq.png")
plt.show()
plt.close()