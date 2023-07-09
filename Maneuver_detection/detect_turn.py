# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:18:58 2021

@author: 91987
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:51:21 2020

@author: 91987
"""
import numpy as np
from matplotlib import pyplot as plt

from scipy import linalg, signal
import scipy.stats as stats
import statsmodels.api as sm
import pandas as pd
import math
from datetime import datetime
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
###########################################################      0000f77c-cb820c98,  0000f77c-62c2a288
def detect_turn(temp):
    #filepath = 'C:\\Users\\91987\\Desktop\\json_info\\locations\\' + '000f8d37-d4c09a0f' + '.txt'
    #filepath = 'C:\\Users\\91987\\Desktop\\json_info\\locations\\' + '0008a165-c48f4b3e' + '.txt'
    #filepath = 'C:\\Users\\91987\\Desktop\\json_info\\locations\\' + '0000f77c-cb820c98' + '.txt'
    #filepath = 'C:\\Users\\91987\\Desktop\\json_info\\location_val\\' + temp + '.json' + '.txt'
    filepath = 'D:\\UAH-DRIVESET-v1\\UAH-DRIVESET-v1\\D5\\20151211170502-16km-D5-DROWSY-SECONDARY\\' + 'RAW_GPS.txt'

    df=pd.read_csv(filepath, index_col=False,sep=' ', header=None)
    print(df[0:10])
    df = df.drop(columns=[4,5,6,7,8,9,10,11,12])
    #df.columns=['time','lat','long','speed']
    df.columns=['time','speed','lat','long']
    
    x = df['time']
    
    y = df['lat']
    z = df['long']
    s = df['speed']
    # y_filter = preprocessor(y)
    # z_filter = preprocessor(z)
    # s_filter = preprocessor(s)
    
    plt.xlabel('Longitude',fontsize= 25)
    plt.ylabel('Latitude',fontsize= 25)
    plt.plot(y,z,linewidth=8,label='Turn')
    plt.legend(prop={'size': 20})
    plt.show()
    
    
    cnt=0
    temp = '20151211170502-16km-D5-DROWSY-SECONDARY' ## only for uah dataset
    
    for i in range(1,len(y)-1):
        phi_prev= y[i-1]
        lamb_prev= z[i-1]
        phi_cur = y[i]
        lamb_cur = z[i]
        phi_next= y[i+1]
        lamb_next= z[i+1]

        del_phi1 = phi_cur - phi_prev
        del_lamb1 = lamb_cur- lamb_prev

        del_phi2 = phi_next - phi_cur
        del_lamb2 = lamb_next- lamb_cur
        sin = 0.707
        cos = 0.707
        res = "No turn"
        val_res = 0
        # cnt=0
        if del_phi1!=0 and del_lamb1!=0 and del_phi2!=0 and del_lamb2!=0:
            cnt = cnt + 1
            m3 = del_phi1/del_lamb1
            m4 = del_phi2/del_lamb2
            theta2 = abs((m3-m4)/(1+m3*m4))
            angle2 = math.degrees(math.atan(theta2))
            print('\nAngle> ', angle2)
            #fp = open('C:\\Users\\91987\\Desktop\\results_turnangles\\sanmateo\\'+ temp + '.csv', 'a')
            fp = open('C:\\Users\\91987\\Desktop\\results_turnangles\\uah\\d5\\'+ temp + '.csv', 'a')

            fp.write(str(temp) + ',' + str(angle2) + '\n')
            #fp.close()
        else:
            #fp = open('C:\\Users\\91987\\Desktop\\results_turnangles\\uah\\d1\\'+ temp + '.csv', 'a')
            fp = open('C:\\Users\\91987\\Desktop\\results_turnangles\\uah\\d5\\'+ temp + '.csv', 'a')

            fp.write(str(temp) + ',' + str(0) + '\n')
            #fp.close()
    fp.close()
            # print(angle2,end="")
            # if(angle2>5):
            #         #print(angle2)
            #         res="Turn detected"
            #         val_res = 1
            #         cnt=cnt+1
            #         fp = open('C:\\Users\\91987\\Desktop\\json_info\\turn_detect_result\\turns'+ '.csv', 'a')
            #         fp.write(str(angle2) + '\n')
            #         fp.close()
            # else:
            #     fp = open('C:\\Users\\91987\\Desktop\\json_info\\turn_detect_result\\noturns'+ '.csv', 'a')
            #     fp.write(str(angle2) + '\n')
            #     fp.close()
        # if del_phi1!=0 or del_phi2!=0 or del_lamb1!=0 or del_lamb2!=0:
        #     del_phi_dash1 = cos* del_phi1 - sin * del_lamb1
        #     del_lam_dash1 = sin* del_phi1 + cos * del_lamb1

        #     del_phi_dash2 = cos* del_phi2 - sin * del_lamb2
        #     del_lam_dash2 = sin* del_phi2 + cos * del_lamb2

        #     m1 = del_phi_dash1*del_lam_dash1
        #     m2 = del_phi_dash2*del_lam_dash2

        #     m3 = del_phi1*del_lamb1
        #     m4 = del_phi2*del_lamb2

        #     theta1 = abs(m1-m2/(1+m1*m2))
        #     theta2 = abs(m3-m4/(1+m3*m4))

        #     angle2 = math.degrees(math.atan(theta2))
        #     if(m1*m2<=0 or m3*m4<=0):
        #         res="Turn detected"
        #     if(angle2>10):
        #         res="Turn detected"

        # print('at', i, res)
        #fp = open('C:\\Users\\91987\\Desktop\\json_info\\turn_detect_result\\' + temp + '.csv', 'a')
        #fp.write(str(i) + ',' + str(val_res) + ',' + res + '\n')
    #fp.close()

    # try:
    #     for i in range(len(x)-1):
    #         #x_del = y_filter[i] - y_filter[i+1]
    #         #y_del = z_filter[i] - z_filter[i+1]
    #         x_del = y[i+1] - y[i]
    #         y_del = z[i+1] - z[i]
    #         # if x_del<0:
    #         #     x_del = -x_del
    #         # if y_del<0:
    #         #     y_del = -y_del
    #         angle = math.atan2(y_del, x_del)*(180/math.pi)
            
    #         print(sec_to_date_time(x[i]),':',i,':','angle',angle)
    #         # if angle<0:
    #         #     angle = angle + 180
    #         # if(angle >= 35):
    #         #     res=" Turn Detected"
    #         # else: res = " No turn"
    #         # print(', '+res)
    # except:
    #     print('Infinite angle')
    #x_del = y[0] - y[1]
    #print(cnt)
    #y_del = z[0] - z[1]
    # print(x_del,y_del)
    # print(math.atan2(y_del, x_del)*(180/math.pi))
    # print(len(x))

def main():
    f1 = open('C:\\Users\\91987\\Desktop\\brakelights_filename.txt', 'r')
    count = 0
    for line in f1:
        line = line.strip()
        temp = line[:-4]
        detect_turn(temp)
        count = count + 1
        print(count, temp)
    f1.close()
if __name__ == '__main__':
    main()
