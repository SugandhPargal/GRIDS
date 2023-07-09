# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 20:20:09 2021

@author: 91987
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 23:07:45 2020

@author: 91987
"""
import sys
import __init__
import numpy as np
from numpy import var, arange
import pandas as pd
from datetime import datetime
import time
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy import linalg, signal
lowess = sm.nonparametric.lowess
#fp = open('C:\\Users\\91987\\Desktop\\results_stops\\.csv','a')

def lowess(x, y, f=2./3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest

    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.

    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations."""
    n = len(x)
    r = int(np.ceil(f*n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:,None] - x[None,:]) / h), 0.0, 1.0)
    w = (1 - w**3)**3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:,i]
            b = np.array([np.sum(weights*y), np.sum(weights*y*x)])
            A = np.array([[np.sum(weights), np.sum(weights*x)],
                   [np.sum(weights*x), np.sum(weights*x*x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1]*x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta**2)**2

    return yest

def preprocessor(sensor):
    x = np.linspace(0,len(sensor),len(sensor))
    b,a = signal.butter(1,0.2)
    bs = signal.filtfilt(b,a,sensor)
    f = 0.2
    return lowess(x,bs,f,3)


limit = 3.0
#ylimit = 5.0
ylimit = 0.1
#zlimit = 10.5
zlimit = 0.01
tlimit = 20.0

def detect_stop(sensory, sensorz):
    #return max(sensory) ## this line is for stops_y
    # fp.write(str(max(sensory)) + ',')
    l = list((filter(lambda z: z <= zlimit, sensorz))) ## this line is for logging the stops_z
    return len(l) ## this line is for logging stops_z
    # # fp.write(str(len(l)) + '\n')
#     if max(sensory) < ylimit:
# 		#if len(filter(lambda z: z >= zlimit, sensorz)) < 3:
#         l = list((filter(lambda z: z <= zlimit, sensorz)))
#         if(len(l) > 3):
#             #print(len(l))
#             return True
        
def call_stopper(filepath, filename):
    df=pd.read_csv(filepath, index_col=False,sep=':')
    print(df.head(10))
    df.columns=['time','valuez','valuey']
    print(df.head(10))
    time=df[['time']].values
    time=time.tolist()
    ts=list()
    ts1=0
    for i in range(len(time)):
        temp= time[i]
        ts1 = sec_to_date_time(temp[0])
        ts.append(ts1)
    #for i in range(len(ts)):  
     #   print(ts[i])
    #print(len(ts))
    #print(time)
    sensorz = df[['valuez']].values
    sensorz = sensorz.tolist()
   # print(sensorz)
    sensorz1=removNestings(sensorz)
    #print(len(sensorz1))
    sensory = df[['valuey']].values
    sensory = sensory.tolist()
    sensory1 = removNestings(sensory)
    
    dict1 = dict()
    for i in range(len(ts)):
        
        if ts[i] not in dict1:
            dict1[ts[i]]= sensorz[i]
        else:
            dict1[ts[i]].append(sensorz[i])
    #print(len(dict1.keys()))
    length = len(dict1.keys())
    keys=sorted(dict1.keys())
    dict2 = dict()
    for i in range(len(ts)):
        
        if ts[i] not in dict2:
            dict2[ts[i]]= sensory[i]
        else:
            dict2[ts[i]].append(sensory[i])
    #print(len(dict2.keys()))
    length = len(dict2.keys())
    keys=sorted(dict2.keys())
    #print(removNestings(dict2[keys[0]]))
    #print(removNestings(dict2[keys[1]]))
    
    #print(sensorz1)
    i=0
    print('******************Stop Detection Results*************')
    
    while(i!=length):
        #print(keys[i])
        #print(i, removNestings(dict1[i]))
        op1=removNestings(dict1[keys[i]])
        opy = removNestings(dict2[keys[i]])
        #print(keys[i]," ", op1, " ",opy)
        #op2=(removNestings(dict1[keys[i+1]]))
            
        #op1.extend(op2)
        if(i==40):
            break
        var_temp = detect_stop(opy, op1)
        fp = open('C:\\Users\\91987\\Desktop\\results_stops\\Z\\sanmateo\\'+ filename + '.csv', 'a')
        fp.write(str(filename) + ',' + str(var_temp) + '\n')
        fp.close()
        # fp.write(filename + ',')   
        # fp.write(str(i+1) + ',')
        #print("++",op1)
        # if(detect_stop(opy, op1)):
            
        #     print(i,'',keys[i],"At timestamp detected stop")
        # else:
            
            
        #     print(i,'',keys[i],"Stop not detected")
        i=i+1
    
    #inset figure
    
    
    #except:
     #   print("index error")
    #print(detect_breaker(sensorz1))'''
        
    # plt.plot(np.arange(len(xy)),preprocessor(xy),linewidth=8,label='Jerkiness')
    # plt.xlabel('Time(s)', fontsize=25, fontweight='bold')
    # plt.ylabel('Acceleration along X-axis',fontsize=25, fontweight='bold')
    # plt.xticks(size=20, fontweight='bold')
    # plt.yticks(size=20, fontweight='bold')
    # plt.legend(prop={'size': 20, 'weight':'bold'}, loc = 'upper right')
    # plt.savefig('C:\\Users\\91987\\Desktop\\landmarks_detect\\maneuvers\\stop.eps',format='eps', bbox_inches='tight')
    # plt.show()
    
    
def sec_to_hour(time):
    hh=int(time/3600)
    mm =int( ((time/float(3600)) - hh)*60)
    ss = int(((((time/float(3600)) - hh)*60) - mm)*60)
    return (str("{:02d}".format(hh))+':'+str("{:02d}".format(mm))+':'+str("{:02d}".format(ss)))

    


def sec_to_date_time(timestamp):
    #print("Helllo")
    #timestamp1 = 1503828254943
    #timestamp2 = 1503828295206
    timestamp1 = int(int(timestamp)/1000)
    #print(timestamp1)
    #timestamp2 = int(timestamp2/1000)
    #print(timestamp2)
    dt_object = datetime.fromtimestamp(timestamp1)
    
    #print("dt_object =", dt_object)
    #print("type(dt_object) =", type(dt_object))
    #dt_object = datetime.fromtimestamp(timestamp2)
    
    #print("dt_object =", dt_object)
    #print("type(dt_object) =", type(dt_object))
    sec = dt_object.strftime("%S")
    mint = dt_object.strftime("%M")
    return (int(sec)+(int(mint)*60))

def removNestings(l): 
    output=list()
    #print(l)
    for i in l: 
        if type(i) == list: 
            z=str(i).strip('[ ]')
            z=z.strip('\'')
            #print(float(z))
            output.append(float(z))
            
        else: 
            output.append(i) 
    #print(output)
    return output
def main():
    file_name = open("C:\\Users\\91987\\Desktop\\brakelights_filename.txt",'r')
    count_line = 1
    
    #path = 'C:\\Users\\91987\\Desktop\\json_info\\yaxis\\' + '0000f77c-6257be58' + '.txt' 
    #path = 'C:\\Users\\91987\\Desktop\\json_info\\yaxis\\' + '0001542f-5ce3cf52' + '.txt' 
    #call_stopper(path)
    for line in file_name:
        #print(len(line.rstrip()))
        #temp = line.rstrip()
        #temp = temp[: -4]
        #print(temp)
        #00268999-cb063914.mov
        line = line.strip()
        line = line[:-4]
        
        path = 'C:\\Users\\91987\\Desktop\\json_info\\z-y_val\\' + line + '.json.txt'
        call_stopper(path, line)
        print('filename' , line)
        
        
        #time.sleep(10)
        
            #count_line = count_line + 1
        count_line = count_line + 1
    #fp.close()
    #path = 'C:\\Users\\91987\\Desktop\\json_info\\yaxis\\' + '0000f77c-62c2a288' + '.txt'
    #call_stopper(path)
    #sec_to_date_time(1503828254943)
    #sec_to_date_time(1503828295206)
if __name__=='__main__':
    main()