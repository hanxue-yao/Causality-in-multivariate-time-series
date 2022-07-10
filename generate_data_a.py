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
    y_arr = np.full((10,4),3.0)
    x_arr_unini = np.full((10,total_length),0.0)
    y_arr_unini = np.full((10,total_length),0.0)
    x = np.hstack((x_arr, x_arr_unini))
    y = np.hstack((y_arr, y_arr_unini))
    
    p = []
    n = [3,3,3,3]
    t1 = [3,3,3,3]
    for i in range(4,total_length+4):  #time
        z.append(math.tanh(z[i-1]+np.random.normal(0,0.01)))
        p.append(z[i]**2+np.random.normal(0,0.05))
        
        for j in range(0,10):
            m = random.randint(1,5)  #random lag
            n = random.randint(1,5)  #random lag
            x[j][i] = sigmoid(z[i-m])+np.random.normal(0,0.01)
            term1 = sigmoid(z[i-m])
            term2 = sigmoid(x[j][i-n])
            noise = np.random.normal(0,1)
            alpha = (abs(term1+term2)/sig_to_noise)/abs(noise)
            
            y[j][i] = term1+term2+alpha*noise


    x=x[:,-total_length:]
    y=y[:,-total_length:]
    p=p[-total_length:]
    z=z[-total_length:]
    
    #add x1 and y1 into dataframe
    #table format 1
    df1=pd.DataFrame({"x1":x[0,:]})
    for m in range(1,10):
        df1.insert(loc=len(df1.columns), column='x'+str(m+1), value=x[m,:])
    for m in range(0,10):
        df1.insert(loc=len(df1.columns), column='y'+str(m+1), value=y[m,:])
    df1.insert(loc=len(df1.columns), column='p', value=p)
    df1.insert(loc=len(df1.columns), column='z', value=z)
    
    '''
    #table format 2
    df1=pd.DataFrame({"x1":x[0,:],"y1":y[0,:]})
    for k in range(1,10):
        df1.insert(loc=len(df1.columns), column='x'+str(k+1), value=x[k,:])
        df1.insert(loc=len(df1.columns), column='y'+str(k+1), value=y[k,:])
        
    df1.insert(loc=len(df1.columns), column='p', value=p)
    df1.insert(loc=len(df1.columns), column='z', value=z)
    '''
    return df1
