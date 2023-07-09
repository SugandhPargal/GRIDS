# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 22:35:12 2020

@author: 91987
"""

import sys
import __init__
import numpy as np
from numpy import var, arange
import pandas as pd
from datetime import datetime
import time as t
from matplotlib import pyplot as plt

import numpy as np
from scipy import linalg, signal
import scipy.stats as stats
import statsmodels.api as sm

lowess = sm.nonparametric.lowess

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

def sec_to_hour(time):
    hh=int(time/3600)
    mm =int( ((time/float(3600)) - hh)*60)
    ss = int(((((time/float(3600)) - hh)*60) - mm)*60)
    return (str("{:02d}".format(hh))+':'+str("{:02d}".format(mm))+':'+str("{:02d}".format(ss)))

    
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

def call_sideslipping(filepath1, filepath2, fp):
    #filepath = 'C:\\Users\\91987\\Desktop\\json_info\\accel-x-y-z\\' + '003baca5-70c87fc6' + '.txt' 

    df=pd.read_csv(filepath1, index_col=False,sep=':')
    #print(df.head(10))
    df.columns=['time','valuex', 'valuey', 'valuez']
    #print(df.head(10))
    time=df[['time']].values
    time=time.tolist()
    ts=list()
    ts1=0
    for i in range(len(time)):
        temp= time[i]
        ts1 = sec_to_date_time(temp[0])
        ts.append(ts1)
    # for i in range(len(ts)):  
    #     print(ts[i])
    # print(len(ts))
    #print(time)
    sensorx = df[['valuey']].values
    sensorx = sensorx.tolist()
    #print(sensorx)
    #t.sleep(10)
   # print(sensorz)
    sensorx1=removNestings(sensorx)
    #print(len(sensorx1),len(ts))
    dict1 = dict()
    for i in range(len(ts)):
        
        if ts[i] not in dict1:
            dict1[ts[i]]= sensorx[i]
        else:
            dict1[ts[i]].append(sensorx[i])
    #print(len(dict1.keys()))
    length = len(dict1.keys())
    keys=sorted(dict1.keys())
    deviation =list()
    for i in range(length):
        op1=removNestings(dict1[keys[i]]) ###### op1 contains accel values(list) for each timestamp ######
        sum1=0
        lensum=0
        for i in op1:
            sum1 = sum1 + i
            lensum = lensum+1
        sum1 =sum1/lensum
        deviation.append(sum1) ## storing avg value for each sec 
    # print(list(dict1.keys()))
    # print(deviation)
   
    # try:
        
    #     plt.xlabel('time')
    #     plt.ylabel('xaxis')
    #     plt.plot(list(dict1.keys()),deviation,label='data+noise')
    #     #plt.show()
    #     plt.xlabel('time')
    #     plt.ylabel('xaxis')
    #     clean_x = preprocessor(deviation)
    #     plt.plot(list(dict1.keys()),clean_x,label='preprocessed')
    #     plt.legend()
    #     plt.title(filepath1)
    #     plt.show()
    #     mean_gyroy = abs(np.mean(clean_x))
    # except np.linalg.LinAlgError as err:
    #     if 'Matrix is singular.' in str(err):
    #         print("handle using raw plot")
            
    #     else:
    #         raise
    # clean_x = preprocessor(deviation)
    # plt.plot(list(dict1.keys()),clean_x,label='preprocessed')
    # plt.legend()
    # plt.title(filepath1)
    # plt.show()
    # mean_gyroy = abs(np.mean(clean_x))
    
    
    ########################################## processing for accelerometer ###########################
    df=pd.read_csv(filepath2, index_col=False,sep=':')
    #print(df.head(10))
    df.columns=['time','valuex', 'valuey', 'valuez']
    #print(df.head(10))
    time=df[['time']].values
    time=time.tolist()
    ts=list()
    ts1=0
    for i in range(len(time)):
        temp= time[i]
        ts1 = sec_to_date_time(temp[0])
        ts.append(ts1)
    # for i in range(len(ts)):  
    #     print(ts[i])
    # print(len(ts))
    #print(time)
    sensorx = df[['valuey']].values
    sensorx = sensorx.tolist()
    #print(sensorx)
    #t.sleep(10)
   # print(sensorz)
    sensorx1=removNestings(sensorx)
    #print(len(sensorx1),len(ts))
    dict1 = dict()
    for i in range(len(ts)):
        
        if ts[i] not in dict1:
            dict1[ts[i]]= sensorx[i]
        else:
            dict1[ts[i]].append(sensorx[i])
    #print(len(dict1.keys()))
    length = len(dict1.keys())
    keys=sorted(dict1.keys())
    deviation2 =list()
    for i in range(length):
        op1=removNestings(dict1[keys[i]]) ###### op1 contains accel values(list) for each timestamp ######
        sum1=0
        lensum=0
        for i in op1:
            sum1 = sum1 + i
            lensum = lensum+1
        sum1 =sum1/lensum
        deviation2.append(sum1) ## storing avg value for each sec
    # plt.xlabel('time')
    # plt.ylabel('xaxis')
    # plt.plot(list(dict1.keys()),deviation2,label='data+noise accel')
    # #plt.show()
    # plt.xlabel('time')
    # plt.ylabel('xaxis')
    try:
        clean_x2 = preprocessor(deviation2)
        clean_x2 = clean_x2.tolist()
    except np.linalg.LinAlgError as err:
        if 'Matrix is singular.' in str(err):
            print("handle using raw plot")
            clean_x2 = deviation2
            
            
        else:
            raise
    # plt.plot(list(dict1.keys()),clean_x2,label='preprocessed')
    # plt.legend()
    # plt.title(filepath1)
    # plt.show()
    mean_accely = abs(np.mean(clean_x2))
    print(clean_x2)
    peak = max(clean_x2)
    peak_index = clean_x2.index(peak)
    filtered_list =  clean_x2
    ############################################ filtering out the list of peak lies within first 5 or last 5 sec of time ################
    if(peak_index >= 0 and peak_index <= 5):
        filtered_list = filtered_list[6:]
        peak = max(filtered_list)
        peak_index = (filtered_list.index(peak))
    elif(peak_index >= 35 and peak_index <= 40):
        filtered_list = filtered_list[:35]
        peak = max(filtered_list)
        peak_index = (filtered_list.index(peak))
    
    print(peak, peak_index)
    std = np.std(filtered_list[peak_index-5: peak_index+6]) ########### standard deviation along the peak ############
    mean = abs(np.mean(deviation[peak_index-5: peak_index+6])) ######### mean value within the lead and trail of the peak ##############
    fp.write(str(peak) + "," +  str(peak_index) + "," + str(std) + "," + 
            str(mean_accely) + "," + "\n") ###### writing to a file for better analyzing the threshold ##############
    ############################################ processing for y-axis values of accelerometer ########
    
   #  df1=pd.read_csv(filepath2, index_col=False,sep=':')
   #  #print(df.head(10))
   #  df1.columns=['time','valuex', 'valuey', 'valuez']
   #  #print(df.head(10))
   #  time=df1[['time']].values
   #  time=time.tolist()
   #  ts=list()
   #  ts1=0
   #  for i in range(len(time)):
   #      temp= time[i]
   #      ts1 = sec_to_date_time(temp[0])
   #      ts.append(ts1)
   #  # for i in range(len(ts)):  
   #  #     print(ts[i])
   #  # print(len(ts))
   #  #print(time)
   #  sensory = df1[['valuey']].values
   #  sensory = sensory.tolist()
   #  #print(sensorx)
   #  #t.sleep(10)
   # # print(sensorz)
   #  sensory1=removNestings(sensory)
   #  #print(len(sensorx1),len(ts))
   #  dict1 = dict()
   #  for i in range(len(ts)):
        
   #      if ts[i] not in dict1:
   #          dict1[ts[i]]= sensory[i]
   #      else:
   #          dict1[ts[i]].append(sensory[i])
   #  #print(len(dict1.keys()))
   #  length = len(dict1.keys())
   #  keys=sorted(dict1.keys())
   #  deviation2y =list()
   #  for i in range(length):
   #      op1=removNestings(dict1[keys[i]]) ###### op1 contains accel values(list) for each timestamp ######
   #      sum1=0
   #      lensum=0
   #      for i in op1:
   #          sum1 = sum1 + i
   #          lensum = lensum+1
   #      sum1 =sum1/lensum
   #      deviation2y.append(sum1) ## storing avg value for each sec
   #  plt.xlabel('time')
   #  plt.ylabel('xaxis')
   #  plt.plot(list(dict1.keys()),deviation2y,label='data+noise accel')
   #  #plt.show()
   #  plt.xlabel('time')
   #  plt.ylabel('xaxis')
   #  clean_x2y = preprocessor(deviation2y)
   #  plt.plot(list(dict1.keys()),clean_x2y,label='preprocessed')
   #  plt.legend()
   #  plt.title(filepath1)
   #  plt.show()
   #  mean_accely_y = abs(np.mean(clean_x2y))
   #  print(clean_x2y)
   #  peak_y = max(clean_x2y)
   #  peak_index_y = clean_x2y.tolist().index(peak_y)
   #  filtered_listy =  clean_x2y.tolist()
   #  ############################################ filtering out the list of peak lies within first 5 or last 5 sec of time ################
   #  if(peak_index_y >= 0 and peak_index_y <= 5):
   #      filtered_listy = filtered_listy[6:]
   #      peak_y = max(filtered_listy)
   #      peak_index_y = (filtered_listy.index(peak_y))
   #  elif(peak_index_y >= 35 and peak_index_y <= 40):
   #      filtered_listy = filtered_listy[:35]
   #      peak_y = max(filtered_listy)
   #      peak_index_y = (filtered_listy.index(peak_y))
    
   #  print(peak_y, peak_index_y)
   #  std_y = np.std(filtered_listy[peak_index_y-5: peak_index_y+6]) ########### standard deviation along the peak ############
   #  mean_y = abs(np.mean(deviation2y[peak_index_y-5: peak_index_y+6]))
    
    df1 = pd.read_csv("C:\\Users\\91987\\Desktop\\sideslipfilename.csv")
    std = df1['std'].tolist()
    mean_accely = df1['accely'].tolist()
    mean_accely_y = df1['accely_y'].tolist()
    for i in range(len(std)):
        
        if(std[i]<=0.007):
            if(std[i]<=0.002):
                if(mean_accely[i]<=0.987):
                    print("0")
                else:
                    if(mean_accely_y[i] <=0.995):
                        print("1")
                    else:
                        if(mean_accely[i] <=0.999):
                            print("0")
                        else:
                            if(std[i]<=0.001):
                                print("1")
                            else:
                                print("0")
            else:
                print("0")
        else:
            print("1")

def call_swerving(filepath1, filepath2, fp):
    #filepath = 'C:\\Users\\91987\\Desktop\\json_info\\accel-x-y-z\\' + '003baca5-70c87fc6' + '.txt' 

    df=pd.read_csv(filepath1, index_col=False,sep=':')
    #print(df.head(10))
    df.columns=['time','valuex', 'valuey', 'valuez']
    #print(df.head(10))
    time=df[['time']].values
    time=time.tolist()
    ts=list()
    ts1=0
    for i in range(len(time)):
        temp= time[i]
        ts1 = sec_to_date_time(temp[0])
        ts.append(ts1)
    # for i in range(len(ts)):  
    #     print(ts[i])
    # print(len(ts))
    #print(time)
    sensorx = df[['valuex']].values
    sensorx = sensorx.tolist()
    #print(sensorx)
    #t.sleep(10)
   # print(sensorz)
    sensorx1=removNestings(sensorx)
    #print(len(sensorx1),len(ts))
    dict1 = dict()
    for i in range(len(ts)):
        
        if ts[i] not in dict1:
            dict1[ts[i]]= sensorx[i]
        else:
            dict1[ts[i]].append(sensorx[i])
    #print(len(dict1.keys()))
    length = len(dict1.keys())
    keys=sorted(dict1.keys())
    deviation =list()
    for i in range(length):
        op1=removNestings(dict1[keys[i]]) ###### op1 contains accel values(list) for each timestamp ######
        sum1=0
        lensum=0
        for i in op1:
            sum1 = sum1 + i
            lensum = lensum+1
        sum1 =sum1/lensum
        deviation.append(sum1) ## storing avg value for each sec 
    # print(list(dict1.keys()))
    # print(deviation)
    plt.xlabel('time')
    plt.ylabel('xaxis')
    plt.plot(list(dict1.keys()),deviation,label='data+noise')
    #plt.show()
    plt.xlabel('time')
    plt.ylabel('xaxis')
    clean_x = preprocessor(deviation)
    plt.plot(list(dict1.keys()),clean_x,label='preprocessed')
    plt.legend()
    plt.title(filepath1)
    plt.show()
    mean_gyroy = abs(np.mean(clean_x))
    
    
    ########################################## processing for accelerometer ###########################
    df=pd.read_csv(filepath2, index_col=False,sep=':')
    #print(df.head(10))
    df.columns=['time','valuex', 'valuey', 'valuez']
    #print(df.head(10))
    time=df[['time']].values
    time=time.tolist()
    ts=list()
    ts1=0
    for i in range(len(time)):
        temp= time[i]
        ts1 = sec_to_date_time(temp[0])
        ts.append(ts1)
    # for i in range(len(ts)):  
    #     print(ts[i])
    # print(len(ts))
    #print(time)
    sensorx = df[['valuex']].values
    sensorx = sensorx.tolist()
    #print(sensorx)
    #t.sleep(10)
   # print(sensorz)
    sensorx1=removNestings(sensorx)
    #print(len(sensorx1),len(ts))
    dict1 = dict()
    for i in range(len(ts)):
        
        if ts[i] not in dict1:
            dict1[ts[i]]= sensorx[i]
        else:
            dict1[ts[i]].append(sensorx[i])
    #print(len(dict1.keys()))
    length = len(dict1.keys())
    keys=sorted(dict1.keys())
    deviation2 =list()
    for i in range(length):
        op1=removNestings(dict1[keys[i]]) ###### op1 contains accel values(list) for each timestamp ######
        sum1=0
        lensum=0
        for i in op1:
            sum1 = sum1 + i
            lensum = lensum+1
        sum1 =sum1/lensum
        deviation2.append(sum1) ## storing avg value for each sec
    plt.xlabel('time')
    plt.ylabel('xaxis')
    plt.plot(list(dict1.keys()),deviation2,label='data+noise accel')
    #plt.show()
    plt.xlabel('time')
    plt.ylabel('xaxis')
    clean_x2 = preprocessor(deviation2)
    plt.plot(list(dict1.keys()),clean_x2,label='preprocessed')
    plt.legend()
    plt.title(filepath1)
    plt.show()
    mean_accely = abs(np.mean(clean_x2))
    
    ###################################################################################################
    #print(filepath1, mean_gyroy, " ", mean_accely)
    peak = max(clean_x2)
    peak_index = clean_x2.tolist().index(peak)
    filtered_list =  clean_x2.tolist()
    ############################################ filtering out the list of peak lies within first 5 or last 5 sec of time ################
    if(peak_index >= 0 and peak_index <= 5):
        filtered_list = filtered_list[6:]
        peak = max(filtered_list)
        peak_index = (filtered_list.index(peak))
    elif(peak_index >= 35 and peak_index <= 40):
        filtered_list = filtered_list[:35]
        peak = max(filtered_list)
        peak_index = (filtered_list.index(peak))
    
    print(peak, peak_index)
    std = np.std(filtered_list[peak_index-5: peak_index+6]) ########### standard deviation along the peak ############
    mean = abs(np.mean(deviation[peak_index-5: peak_index+6])) ######### mean value within the lead and trail of the peak ##############
    fp.write(str(peak) + "," +  str(peak_index) + "," + str(std) + "," + 
          str(mean_gyroy) + "," +  str(mean_accely) + ",") ###### writing to a file for better analyzing the threshold ##############
    
    
    print(std, mean)
    
    if((mean_gyroy>=0.01 and mean_gyroy<=0.055) and ########## detect swerving #################
    (mean_accely>=0.98 and mean_accely<=1.01) and
    (std>=0.001 and std<=0.004)):
        if(abs(peak)>=0.98 and abs(peak)<=1.007):
            print("swerving")
            fp.write("1" + "\n")
        else:
            print("not swerving")
            fp.write("0" + "\n")
    elif(not(mean_gyroy>=0.01 and mean_gyroy<=0.055) and 
    (mean_accely>=0.98 and mean_accely<=1.01)):
        
        if((mean_gyroy>=0.0003 and mean_gyroy<=0.001)
           and abs(peak)>=0.98 and abs(peak)<=1.007):
            print("swerving")
            fp.write("1" + "\n")
        else:
            print("not swerving")
            fp.write("0" + "\n")
    elif((mean_accely>=0.98 and mean_accely<=1.01) and
    not(std>=0.001 and std<=0.004)):
        
        if((std>=0.0005 and std<=0.004)
           and abs(peak)>=0.98 and abs(peak)<=1.007):
            print("swerving")
            fp.write("1" + "\n")
        else:
            print("not swerving")
            fp.write("0" + "\n")
        
    else:

        print("not swerving")
        fp.write("0" + "\n")
        
    
    
    

def call_weaving(filepath, fp):
    #filepath = 'C:\\Users\\91987\\Desktop\\json_info\\accel-x-y-z\\' + '003baca5-70c87fc6' + '.txt' 

    df=pd.read_csv(filepath, index_col=False,sep=':')
    #print(df.head(10))
    df.columns=['time','valuex', 'valuey', 'valuez']
    #print(df.head(10))
    time=df[['time']].values
    time=time.tolist()
    ts=list()
    ts1=0
    for i in range(len(time)):
        temp= time[i]
        ts1 = sec_to_date_time(temp[0])
        ts.append(ts1)
    # for i in range(len(ts)):  
    #     print(ts[i])
    # print(len(ts))
    #print(time)
    sensorx = df[['valuex']].values
    sensorx = sensorx.tolist()
    #print(sensorx)
    #t.sleep(10)
   # print(sensorz)
    sensorx1=removNestings(sensorx)
    print(len(sensorx1),len(ts))
    #t.sleep(10)
    
    # sensory = df[['valuey']].values
    # sensory = sensory.tolist()
    # sensory1 = removNestings(sensory)
    dict1 = dict()
    for i in range(len(ts)):
        
        if ts[i] not in dict1:
            dict1[ts[i]]= sensorx[i]
        else:
            dict1[ts[i]].append(sensorx[i])
    #print(len(dict1.keys()))
    length = len(dict1.keys())
    keys=sorted(dict1.keys())
    deviation =list()
    for i in range(length):
        op1=removNestings(dict1[keys[i]]) ###### op1 contains accel values(list) for each timestamp ######
        # if(i==33):
        #     xx = op1
        #     print(xx)
        # elif(i==11):
        #     xx1 = op1
        #     print('10', xx1)
        # elif(i==34):
        #     xx2 = op1
        #     print('11', xx2)
        
        sum1=0
        lensum=0
        for i in op1:
            sum1 = sum1 + i
            lensum = lensum+1
        sum1 =sum1/lensum
        deviation.append(sum1) ## storing avg value for each sec 
    # tt = range(len(xx))
    # print(len(xx),len(tt))    
    # plt.xlabel('Time', fontsize=25)
    # plt.ylabel('Acceleration along X-axis',fontsize=15)
    # plt.plot(tt,preprocessor(xx),linewidth=8,label='Normal')
    # plt.legend(prop={'size': 10})
    # plt.show()
    # print('*************************************************')
    # print(xx)
    
    # tt = range(len(xx1))
    # print(len(xx1),len(tt))    
    # plt.xlabel('Time', fontsize=25)
    # plt.ylabel('Acceleration along X-axis',fontsize=18)
    # plt.plot(tt,preprocessor(xx1),'orange',linewidth=8,label='Lane-Changes')
    # plt.legend(prop={'size': 15}, loc = 'upper left')
    # plt.show()
    # print('**************************************************')
    # print(xx1)   
    # # xx.extend(xx1)
    # # xx.extend(xx2)
    # # tt = range(len(xx))
    # # print(len(xx),len(tt))    
    # # plt.xlabel('Time', fontsize=25)
    # # plt.ylabel('Acceleration along X-axis',fontsize=15)
    # # plt.plot(tt,preprocessor(xx),linewidth=8,label='Normal')
    # # plt.legend(prop={'size': 20}, loc= 'best')
    # # plt.show()
        
    # fig = plt.figure()
    # axes1 = fig.add_axes([0.1, 0.1, 0.9, 0.9]) # main axes
    # axes2 = fig.add_axes([0.3, 0.4, 0.5, 0.4])# inset axes
    # # main figure
    # axes1.plot(np.arange(len(xx1)), preprocessor(xx1),linewidth=8,label='Normal')
    # axes1.set_xlabel('Time(ms)',fontsize=25)
    # axes1.set_ylabel('Acceleration along Z-axis',fontsize=25)
    # axes1.legend(prop={'size': 10})
    # axes1.set_title('',fontsize=25)
    
    # # insert
    # axes2.plot(np.arange(len(xx)), xx,'orange',linewidth=8,label='Lane-Change')
    # axes2.set_xlabel('',fontsize=20)
    # axes2.set_ylabel('',fontsize=20)
    # axes2.set_xticklabels(axes2.get_xticks(), {'weight':'bold'})
    # axes2.set_yticklabels(axes2.get_yticks(), {'weight':'bold'})
    # axes2.legend(prop={'size': 15})
    # axes2.set_title('',fontsize=20)

    #dict2 = dict()
    # for i in range(len(ts)):
        
    #     if ts[i] not in dict2:
    #         dict2[ts[i]]= sensory[i]
    #     else:
    #         dict2[ts[i]].append(sensory[i])
    # print(len(dict2.keys()))
    # length = len(dict2.keys())
    # keys=sorted(dict2.keys())
    print(list(dict1.keys()))
    print(deviation)
    plt.xlabel('time')
    plt.ylabel('xaxis')
    plt.plot(list(dict1.keys()),deviation,label='data+noise')
    #plt.show()
    plt.xlabel('time')
    plt.ylabel('xaxis')
    clean_x = preprocessor(deviation)
    plt.plot(list(dict1.keys()),clean_x,label='preprocessed')
    plt.legend()
    plt.title(filepath)
    plt.show()
    
    ############# writing logic to detect weaving #######################
    peak = max(deviation)
    trail = min(deviation)
    peak_index = deviation.index(peak)
    trail_index = deviation.index(trail)
    range_val = abs(peak - trail)
    duration = abs(trail_index - peak_index)
    
    if(peak_index<=trail_index):
        std = np.std(deviation[peak_index: trail_index+1])
        mean = abs(np.mean(deviation[peak_index: trail_index+1]))
    else:
        std = np.std(deviation[trail_index: peak_index+1])
        mean = abs(np.mean(deviation[trail_index: peak_index+1]))
    print("peak = ", peak, "trail = ", trail, "peak_index = ", peak_index,
          "trail_index = ", trail_index, "range_val = ", range_val, "duration = ", duration,
          "standard deviation = ", std, "mean = ", mean)
    fp.write(str(peak) + "," +  str(trail) + "," + str(peak_index) + "," + 
          str(trail_index) + "," +  str(range_val) + "," + str(duration) + "," + 
          str(std) + "," + str(mean) + ",")
    if(range_val>=0.02 and range_val<=0.18 and std>=0.01 and std<=0.05 and
        mean>=0.97 and mean<=1.02):
        if(duration<=10):
            print("weaving present")
            fp.write("1" + "\n")
        else:
            print("weaving not present")
            fp.write("0" + "\n")
    else:
            print("weaving not present")
            fp.write("0" + "\n")
        
    
    # lst = list()
    # for j in range(length-1):
    #     print(j,'',clean_x[j], '',deviation[j],'\n')
    #     print(round(abs(deviation[j]-deviation[j+1]),3))
    #     lst.append(round(abs(deviation[j]-deviation[j+1]), 3))
    # print(removNestings(dict1[keys[0]]))
    # print(removNestings(dict1[keys[1]]))
    # lst.append(0)
    # plt.xlabel('time')
    # plt.ylabel('xaxis')
    # plt.plot(list(dict1.keys()),lst,label='clean')
    # plt.show()
    


def main():
    file_name = open("C:\\Users\\91987\\Desktop\\brakelights_filename.txt",'r')
    fp = open("C:\\Users\\91987\\Desktop\\sideslipfilename.csv",'a')
    #fp.write("filename" + "," + "peak" + "," +  "trail" + "," + "peak_index" + "," + 
    #       "trail_index" + "," +  "range_val" + "," + "duration" + "," + 
    #       "standard deviation" + "," + "mean" + "," + "weaving" + "\n")
    fp.write("filename" + "," + "peak" + "," +  "peak_index" + "," + "std" + "," + "accely" + "," + "side" + "\n")
    
    #count_line = 1
    for i in file_name:
        #temp = i[]
        temp = i[:-5]
        print(temp)
        fp.write(temp + ",")
        
        path1 = 'C:\\Users\\91987\\Desktop\\json_info\\gyro-x-y-z\\' + temp + '.txt' 
        path2 = 'C:\\Users\\91987\\Desktop\\json_info\\accel-x-y-z\\' + temp + '.txt'
        call_sideslipping(path1, path2, fp)
    fp.close()
    
    
if __name__=='__main__':
    main()
#################################################################  temp processing ##################################
import pandas as pd
df = pd.read_csv('C:\\Users\\91987\\Desktop\\file_temp.csv', index_col=False)
actual_value = df['col6']
predicted_value = df['col12']


import numpy as np
#actual_value = np.array([0.0090006939,0.0131126625,0.01970882295,0.01070812905,0.0131126625,0,0.0886568563,0.0886568563])
#predicted_value = np.array([0.008598175164,0.01163257545,0.01616202397,0.007563848802,0.01163257545,0.004896300226,0.03561509511,0.03071879488])

# take square of differences and sum them
l2 = np.sum(np.power((actual_value-predicted_value),2))
print(l2)