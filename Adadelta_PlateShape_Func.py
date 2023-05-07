import math as math
import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from scipy.misc import derivative  
Rho = 0.9  
Epsy = pow(10 , -5) 
X1 , X2 = smp.symbols('X1  X2 ' , real = True ) 
f = (X1+2*X2-7)**2 + (2*X1+X2-5)**2   #Booth Func
dfdx_1 = smp.diff(f , X1) 
dfdx_2 = smp.diff(f , X2)  
learning_rate = 0.0001
Stop = 3
Amp_df = 3.2
dx_1 = 0.00
dx_2 = 0.00
iteration = 0  
delta_X_previous = ([0.0],
                          [0.0])    
delta_X_current = ([0.0] 
                  ,[0.0])  
current = ([0.0] , 
           [0.0]) 
previous = ([3.0] , 
            [1.50])
S_Vector_previous =  ([[0.0] 
                           ,[0.0]]) 
S_Vector_current = ([0.0],
                          [0.0])    
gradient_dash = ([0.0] 
                  ,[0.0]) 
Epsy_vector = ([Epsy] , 
               [Epsy])
arr_plot = np.array([])  
time = 1 
while Amp_df > Stop  :  
    time +=1 
    iteration += 1 
    dx_1 = int(dfdx_1.subs([(X1 ,previous[0][0] ) , (X2 , previous[1][0])])) 
    dx_2 = int(dfdx_2.subs([(X1 , previous[0][0] ) , (X2 , previous[1][0])]))
    gradient = ([[dx_1]
                 ,[dx_2]])
    Amp_df = math.sqrt(((dx_1)**2)+((dx_2)**2)) 
    S_Vector_current  = np.add(np.multiply(Rho , S_Vector_previous ) , np.multiply((1-Rho) , np.square(gradient)))  
    gradient_dash = np.multiply (np.divide(np.sqrt(np.add(delta_X_previous , Epsy_vector)) , np.sqrt(np.add(S_Vector_current , Epsy_vector))) , gradient ) 
    delta_X_current =np.add(np.multiply(Rho , delta_X_previous ) ,  np.multiply((1-Rho) , np.square(gradient_dash))) 
    current = np.subtract(previous  , gradient_dash ) 
    previous = current    
    delta_X_previous = delta_X_current   
    arr_plot = np.append(arr_plot , Amp_df )
    print (Amp_df) 
x = range(iteration)
y = arr_plot
plt.plot(x,y)
plt.show() 
print(iteration)