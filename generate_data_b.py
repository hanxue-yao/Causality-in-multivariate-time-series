import math
import numpy as np
import pandas as pd
import random


total_length=1000

def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
def generate(sig_to_noise):
    np.random.seed(0)
    z = [1,1,1,1]
    x_arr = np.full((10,4),1.0)
    x_arr_unini = np.full((10,total_length),0.0)
    x = np.hstack((x_arr, x_arr_unini))
    
    p = []
    n = [3,3,3,3]
    t1 = [3,3,3,3]
    for i in range(4,total_length+4):   #time
        z.append(math.tanh(z[i-1]+np.random.normal(0,0.01)))
        p.append(z[i]**2+np.random.normal(0,0.05))
        
        for j in range(0,10):
            m = random.randint(1,5)  #random lag
            n = random.randint(1,5)  #random lag
            term1 = sigmoid(z[i-m])
            term2 = sigmoid(x[j][i-n])
            noise = np.random.normal(0,1)
            alpha = (abs(term1+term2)/sig_to_noise)/abs(noise)
                
            x[j][i] = term1+term2+alpha*noise



    x=x[:,-total_length:]
    p=p[-total_length:]
    z=z[-total_length:]
    
     #table format 1
    df2=pd.DataFrame({"x1":x[0,:]})
    for m in range(1,10):
        df2.insert(loc=len(df2.columns), column='x'+str(m+1), value=x[m,:])
    
    
    df2.insert(loc=len(df2.columns), column='p', value=p)
    df2.insert(loc=len(df2.columns), column='z', value=z)
   
    return df2
