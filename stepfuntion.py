

import numpy as np
from matplotlib import pyplot as plt
import seaborn
w=-3.2
t=0.5

def stufe(z,t):
  if(z <= t):
    return 0
  else:
    return 1

x_arr = np.arange(start=-10, stop=11,step=0.01)
y_list = []

index = 0
for x in x_arr:
    # Ausgabe des Neurons berechnen und an Liste y anhaengen
    z = w*x
    if(z >t and x_arr[index + 1] * w <= t): # sprung
      print(x) 
    index+=1
    y_list.append(stufe(z,t))

plt.figure(figsize=(16, 6))
plt.plot(x_arr, y_list)
plt.xlabel("Eingabe x")
plt.ylabel("Ausgabe y")
plt.grid(True)
plt.show()

