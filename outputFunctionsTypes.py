%matplotlib inline
# erlaubt das partielle Definieren von Funktionsargumenten
from functools import partial

# numerische Berechnungen
import numpy as np



def slow_leaky_relu_backward(z, alpha=0.01):
    assert z.ndim == 1, "Nur fuer 1D-Eingaben konzipiert"
    dydz = np.empty_like(z)
    for i, z_i in enumerate(z):
        # Ausgabe fuer jedes Element z_i bestimmen
        if z_i <= 0:
            dydz[i] = alpha
        else:
            dydz[i] = 1
    return dydz

def fast_leaky_relu_backward(z, alpha=0.01):
    # betrachtete Faelle schliessen sich aus -> Realisierung durch Addition moeglich
    dydz = (z <= 0) * alpha + (z > 0) * 1
    return dydz

def even_faster_leaky_relu_backward(z, alpha=0.01):
    # Default-Ausgabewert: 1
    dydz = np.ones_like(z)
    # Teile in Ausgabe korrigieren
    dydz[z <= 0] = alpha
    return dydz


# Funktionsausgaben visualisieren -> Werteverlaufgleichheit pruefen
plot(forward_functions=(),
     backward_functions=(slow_leaky_relu_backward,
                         fast_leaky_relu_backward,
                         even_faster_leaky_relu_backward),
     labels=('slow', 'fast', 'even faster'),
     linestyles=('-', '--', ':'))

# Ausfuehrung timen und vergleichen
z = np.linspace(-8, 8, 1000)
%timeit -n 100 slow_leaky_relu_backward(z)
%timeit -n 100 fast_leaky_relu_backward(z)
%timeit -n 100 even_faster_leaky_relu_backward(z)


def sigmoid_forward(z):
    return 1 / (1 + np.exp(-z))     

def sigmoid_backward(z):
    return  (sigmoid_forward(z)) * (1 - (sigmoid_forward(z)))  
plot(forward_functions=sigmoid_forward,
     backward_functions=sigmoid_backward,
     labels='Sigmoid')
def tanh_forward(z):
    return np.tanh(z)    

def tanh_backward(z):
    return 1 - (np.tanh(z) ** 2)    
 
# Funktionsverlauf visualisieren
plot(forward_functions=tanh_forward,
     backward_functions=tanh_backward,
     labels='Tanh')
