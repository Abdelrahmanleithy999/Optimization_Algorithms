import math as math
import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from scipy.misc import derivative 
from decimal import Decimal 
X1 , X2  , η = smp.symbols('X1 X2  η', real=True)   
f = (1.5 - X1 + X1*X2)**2 + (2.25-X1 + X1*((X2)**2))**2 + (2.625-X1+X1*((X2)**3))**2  # BEALE FUNC  3 , 0.5 
local_min = ([[3] , 
              [0.5]])  
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
arr_plot = np.array([]) # for drawing curve 
while Amp_df > Stop :
    iteration += 1
    dx_1 = int(dfdx_1.subs([(X1 , previous[0][0]) , (X2 , previous[1][0])]))
    dx_2 = int(dfdx_2.subs([(X1 , previous[0][0]) , (X2 , previous[1][0])])) 
    Amp_df = Decimal(math.sqrt((((dx_1)**2)+((dx_2)**2))))
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
print("accuracy of X1 is ")
print((((local_min[0][0]) - (current[0][0]))/(local_min[0][0]))*100)
print("accuracy of X2 is ") 
print((((local_min[1][0]) - (current[1][0]))/(local_min[1][0]))*100) 
