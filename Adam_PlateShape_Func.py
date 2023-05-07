import math as math
import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from scipy.misc import derivative 
Alpha = 0.001 
Beta_1 = 0.9 
Beta_2 = 0.999 
Epsy = pow(10 , -8)
X1 , X2 = smp.symbols('X1  X2 ' , real = True ) 
f = (X1+2*X2-7)**2 + (2*X1+X2-5)**2   #Booth Func
local_min = ([[1] , 
              [3]]) 
dfdx_1 = smp.diff(f , X1) 
dfdx_2 = smp.diff(f , X2)  
learning_rate = 0.001
Stop = 0.001
Amp_df = 3.2
dx_1 = 0.00
dx_2 = 0.00
iteration = 0  
current = ([0.0] , 
           [0.0]) 
previous = ([1] , 
            [0.5])
moment_vector_current =  ([[0.0] 
                           ,[0.0]]) 
moment_vector_previous = ([1],
                          [0.5])   
moment_vector_new =  ([0.0],
                       [0.0]) 
                       
velocity_current =  0.0 
                           
velocity_previous =  0.0
                           
velocity_new = 0.0 
g = ([[0.0] , 
      [0.0]])  
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
    moment_vector_current  = np.add(np.multiply(Beta_1 , moment_vector_previous) ,  np.multiply((1-Beta_2) , gradient))
    velocity_current = (Beta_2)*(velocity_previous) +  (1-Beta_2)*((g[0][0]) + (g[1][0])) 
    moment_vector_new =  np.divide(moment_vector_current ,(1-(pow(Beta_1 , time)))) 
    velocity_new  = (velocity_current/(1-(pow(Beta_2 , time))))
    custimized_eta = ((learning_rate)/(math.sqrt(velocity_new+Epsy)))
    current = np.subtract (previous , np.multiply(custimized_eta ,moment_vector_new ))
    moment_vector_previous = moment_vector_current
    velocity_previous = velocity_current
    previous = current   
    arr_plot = np.append(arr_plot , Amp_df )
    print (Amp_df) 
x = range(iteration)
y = arr_plot
plt.plot(x,y)
plt.show() 
print(iteration)
print(current)