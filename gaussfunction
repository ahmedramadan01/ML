import numpy as np
from matplotlib import pyplot as plt
import seaborn
w = 1.3
sigma = 1.6

def gauss(z,sigma):
  y = np.exp(-z**2/2*sigma**2)
  return y

def gauss_d(z,y):
    gd = -(1/sigma**2)*z*y
    return gd

def gauss_dd(z,sigma):
  gdd = (z**2 - sigma**2) * np.exp((-z**2)/(2*sigma**2)) / (sigma**2 * sigma**2)
  return gdd

x_arr = np.arange(start=-10, stop=11,step=0.1)

y_list = []
y_list_d=[]
y_list_dd = []
# für alle Elemente in x_arr:
for x in x_arr:
    # Ausgabe des Neurons berechnen und an Liste y anhaengen
    z = w*x
    y = gauss(z,sigma=sigma)
    y_list.append(gauss(z, sigma=sigma))
    y_list_d.append(gauss_d(z,y))
    y_list_dd.append(gauss_dd(z,sigma))




plt.figure(figsize=(16, 6))
plt.plot(x_arr, y_list)
plt.plot(x_arr,y_list_d)
plt.plot(x_arr,y_list_dd)
plt.xlabel("Eingabe x")
plt.ylabel("Ausgabe y")
plt.grid(True)
plt.show()
