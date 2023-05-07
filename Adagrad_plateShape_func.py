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
f = (X1+2*X2-7)**2 + (2*X1+X2-5)**2   #Booth Func
dfdx_1 = smp.diff(f , X1) 
dfdx_2 = smp.diff(f , X2)  
learning_rate = 0.00001
Stop = 3
Amp_df = 3.2
dx_1 = 0.00
dx_2 = 0.00
iteration = 0  
current = ([0.0] , 
           [0.0]) 
previous = ([3.0] , 
            [2.0])
Adagrad_Vector_previous =  ([[3.0] 
                           ,[2.0]]) 
Adagrad_Vector_current = ([0.0],
                          [0.0])    
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
    Adagrad_Vector_current = np.add(Adagrad_Vector_previous , np.square(gradient))  
    value_1 = np.multiply(Epsy , gradient )  
    value_2 = np.add(np.sqrt(Adagrad_Vector_current) , Epsy_2)  
    value_3 = np.divide(value_1 , value_2)  
    current = np.subtract (previous , value_3) 
    Adagrad_Vector_previous = Adagrad_Vector_current 
    previous = current   
    arr_plot = np.append(arr_plot , Amp_df )
    print (Amp_df) 
x = range(iteration)
y = arr_plot
plt.plot(x,y)
plt.show() 
print(iteration) 
print(current)