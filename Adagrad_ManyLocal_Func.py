import math as math
import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from scipy.misc import derivative 
Alpha = 0.001 
Beta_1 = 0.9 
Beta_2 = 0.999 
Epsy = pow(10 , -3) 
Epsy_2 = pow(10 , -6) 
X1 , X2 = smp.symbols('X1  X2 ' , real = True ) 
f = 0.5 + (((smp.sin((X1**2) - (X2**2)))**2)  - 0.5 )/(1+0.001*(X1**2 + X2**2))**2   #schaffer function
dfdx_1 = smp.diff(f , X1) 
dfdx_2 = smp.diff(f , X2)  
learning_rate = 0.001
Stop = 3
Amp_df = 3.2
dx_1 = 0.00
dx_2 = 0.00
iteration = 0  
current = ([0.0] , 
           [0.0]) 
previous = ([3.0] , 
            [2.0]) 
g = ([0.0] , 
     [0.0]) 
Adagrad_previous =  0.0  
                           
Adagrad_current =  0.0 
                              
arr_plot = np.array([])  
time = 1 
while Amp_df > Stop  :  
    time +=1 
    iteration += 1 
    dx_1 = int(dfdx_1.subs([(X1 ,previous[0][0] ) , (X2 , previous[1][0])])) 
    dx_2 = int(dfdx_2.subs([(X1 , previous[0][0] ) , (X2 , previous[1][0])]))
    gradient = ([[dx_1]
                 ,[dx_2]]) 
    g = np.square(gradient) 
    Amp_df = math.sqrt(((dx_1)**2)+((dx_2)**2)) 
    Adagrad_current = Adagrad_previous + (g[0][0])+(g[1][0])  
    value_1 =  learning_rate    
    value_2 =    math.sqrt(Adagrad_current + Epsy_2)  
    Custimized_Eta = value_1 / value_2   
    value_3 = np.multiply(Custimized_Eta , gradient)
    current = np.subtract (previous , value_3)
    Adagrad_previous = Adagrad_current 
    previous = current   
    arr_plot = np.append(arr_plot , Amp_df )
    print (Amp_df) 
x = range(iteration)
y = arr_plot
plt.plot(x,y)
plt.show() 
print(iteration) 
print(current)