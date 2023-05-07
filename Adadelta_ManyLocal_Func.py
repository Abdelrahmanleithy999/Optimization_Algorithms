import math as math
import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from scipy.misc import derivative  
Rho = 0.9  
Epsy = pow(10 , -5) 
X1 , X2 = smp.symbols('X1  X2 ' , real = True ) 
f = 0.5 + (((smp.sin((X1**2) - (X2**2)))**2)  - 0.5 )/(1+0.001*(X1**2 + X2**2))**2   #schaffer function
dfdx_1 = smp.diff(f , X1) 
dfdx_2 = smp.diff(f , X2)  
learning_rate = 0.0001
Stop = 0.0001
Amp_df = 3.2
dx_1 = 0.00
dx_2 = 0.00
iteration = 0  
delta_X_previous =  0.0
delta_X_current =  0.0
current = ([0.0] , 
           [0.0]) 
previous = ([1] , 
            [0.5])
S_previous =  0.0 
S_current =  0.0 
gradient_dash = ([0.0] 
                  ,[0.0]) 
arr_plot = np.array([])  
g = ([0.0],
     [0.0]) 
g_2 = ([0.0],
     [0.0])
time = 1 
while Amp_df > Stop  :  
    time +=1 
    iteration += 1 
    dx_1 = int(dfdx_1.subs([(X1 ,previous[0][0] ) , (X2 , previous[1][0])])) 
    dx_2 = int(dfdx_2.subs([(X1 , previous[0][0] ) , (X2 , previous[1][0])]))
    gradient = ([[dx_1]
                 ,[dx_2]])
    g = np.square(gradient)  
    g_2 = np.square(gradient_dash)
    Amp_df = math.sqrt(((dx_1)**2)+((dx_2)**2)) 
    S_current  = Rho*S_previous + (1-Rho)*(g[0][0])+(g[1][0])  
    delta_X_current = (Rho)*(delta_X_previous)+((1-Rho)*((g_2[0][0]) + (g_2[1][0])))   
    value = (math.sqrt(delta_X_previous + Epsy))/(math.sqrt(S_current + Epsy))
    gradient_dash = np.multiply ( value , gradient ) 
    current = np.subtract(previous  , gradient_dash ) 
    previous = current    
    delta_X_previous = delta_X_current    
    S_previous = S_current
    arr_plot = np.append(arr_plot , Amp_df )
    print (Amp_df) 
x = range(iteration)
y = arr_plot
plt.plot(x,y)
plt.show() 
print(iteration)
print(current)