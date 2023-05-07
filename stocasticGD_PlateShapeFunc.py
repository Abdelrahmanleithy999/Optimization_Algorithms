import math as math
import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from scipy.misc import derivative 
X1 , X2  , η = smp.symbols('X1 X2  η', real=True)  
f = (X1+2*X2-7)**2 + (2*X1+X2-5)**2   #Booth Func  DONE  local minmum = 1 , 3  
local_min = ([[1] , 
              [3]])  
dfdx_1 = smp.diff(f, X1) 
dfdx_2 = smp.diff(f, X2)  
learning_rate = 0.001
Stop = 0.001 
Amp_df = 3.2
dx_1 = 0.00
dx_2 = 0.00
iteration = 0 
current = ([[0.00],
            [0.00]
            ])
previous = ([[1],
             [0.5]
             ])
arr_plot = np.array([])
while Amp_df > Stop :
    iteration += 1
    dx_1 = int(dfdx_1.subs([(X1 , previous[0][0]) , (X2 , previous[1][0])]))
    dx_2 = int(dfdx_2.subs([(X1 , previous[0][0]) , (X2 , previous[1][0])])) 
    Amp_df = math.sqrt((((dx_1)**2)+((dx_2)**2)))
    g_df = ([[dx_1],
             [dx_2]
             ])
    current = np.subtract(previous,(np.multiply(learning_rate,g_df)))
    arr_plot = np.append(arr_plot , Amp_df)
    previous = current
    print (Amp_df)

x = range(iteration)
y = arr_plot
plt.plot(x,y)
plt.show() 
print(iteration) 
print(current) 

