import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('google.csv')
df['Open'].plot()

def make_data(df,win):
    k = 0
    x = []
    y = []
    while (k+win<len(df)-1):
        xa = df['Open'][k:k+win]
        ya = df['Open'][k+win+1]
        x.append(xa)
        y.append(ya)
        k+=1
    
    x = np.array(x)
    y = np.array(y)
    
    return x,y

def split_data(x,y):
    return train_test_split(x,y,test_size = 0.2,shuffle = False)


x,y = make_data(df,10)
train_x,test_x,train_y,test_y = split_data(x,y)
print(train_x.shape)
print(test_y.shape)
print(train_y.shape)
print(test_x.shape)