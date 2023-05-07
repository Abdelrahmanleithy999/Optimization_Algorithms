import math as math
import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from scipy.misc import derivative 
momentum = 0.9 
X1 , X2  , η = smp.symbols('X1 X2  η', real=True)  
f = 2*X1**2 - 1.05*X1**4 + (X1**6)/6 + X1*X2 + X2**2  #Three Hump Camel Function 
dfdx_1 = smp.diff(f, X1) 
dfdx_2 = smp.diff(f, X2)  
learning_rate = 0.001
Stop = 0.001
Amp_df = 3.2
dx_1 = 0.00
dx_2 = 0.00
iteration = 0 
current_Velocity = ([[0.00],
            [0.00]
            ])
previous_Velocity = ([[1.0],
             [0.5]
               ])
current = ([[0.00],
            [0.00]
            ])
previous = ([[1.0],
             [0.5]
             ])
arr_plot = np.array([])
while Amp_df > Stop :
    ##velocity = momentum * velocity - learning_rate * g
#w = w + velocity
    iteration += 1
    dx_1 = int(dfdx_1.subs([(X1 , previous[0][0]) , (X2 , previous[1][0])]))
    dx_2 = int(dfdx_2.subs([(X1 , previous[0][0]) , (X2 , previous[1][0])])) 
    Amp_df = math.sqrt((((dx_1)**2)+((dx_2)**2)))
    g_df = ([[dx_1],
             [dx_2]
             ]) 
    current_Velocity = np.add((np.multiply(momentum,previous_Velocity)) , (np.multiply(learning_rate,g_df)))
    current = np.subtract(previous,current_Velocity)
    arr_plot = np.append(arr_plot , Amp_df) 
    previous_Velocity = current_Velocity  
    previous = current 
    print (Amp_df)

x = range(iteration)
y = arr_plot
plt.plot(x,y)
plt.show() 
print(current)
print(iteration)