import numpy as np
from matplotlib import pyplot as plt
import seaborn

w= -0.8
c= 0.8

def fermi(z,c,x):
  y = 1/(1 + np.exp(-c*z))
  return y


def fermi_derivative(z,c,x):
    f = fermi(z,c,x)
    df = f*(1-f)
    return df

x_arr = np.arange(start=-10, stop=11,step=0.01)

print(f"Die ersten zehn Elemente des x-Vektors:\n{x_arr[0:21]}\n")

print(len(x_arr))

y_list = []
y_list_d=[]
# für alle Elemente in x_arr:
for x in x_arr:
    # Ausgabe des Neurons berechnen und an Liste y anhaengen
    z = w*x
    y_list.append(fermi(z, c=c,x=x))
    y_list_d.append(fermi_derivative(z,c=c,x=x))

print(y_list)
print(len(y_list))


plt.figure(figsize=(16, 6))
plt.plot(x_arr, y_list)
plt.plot(x_arr,y_list_d)
plt.xlabel("Eingabe x")
plt.ylabel("Ausgabe y")
plt.grid(True)
plt.show()


