import numpy as np
def quartiles(feature):
    Q1 = np.percentile(feature, 25, interpolation = 'midpoint')
    Q2 = np.percentile(feature, 50, interpolation = 'midpoint')
    Q3 = np.percentile(feature, 75, interpolation = 'midpoint')
    IQR = Q3-Q1
    Q4 = np.percentile(feature, 85, interpolation = 'midpoint')
    return [Q1,Q2,Q3,IQR,Q4]



def l2_norm(vec1, vec2):
    val = np.sqrt((vec1[0])**2 + (vec1[1])**2 + (vec1[2])**2 + (vec1[3])**2) ### l2_norm for individual feature between quartiles
    return val

    
def rank_feature(f1, f2, f3, f4, f5, f6, f7, f8): #### 4 categories list ## f1,f3,f5,f7 for home city
                                                    #### f2, f4, f6, f8 for visiting city
    quart_f1 = []
    quart_f2 = []
    quart_f3 = []
    quart_f4 = []
    l2_f1 = 0
    l2_f3 = 0
    l2_f5 = 0
    l2_f7 = 0
    
    
    for i in range(len(f1)):
        quart_f1_h = quartiles(f1[i]) ##home city quartiles feature wise
        quart_f1_v = quartiles(f2[i]) ## visiting city quartiles feature wise
        l2_f1 += l2_norm(quart_f1_h, quart_f1_v) ##home city l2-norm feature wise
        
    for i in range(len(f3)):
        quart_f3_h = quartiles(f3[i]) ##home city quartiles feature wise
        quart_f3_v = quartiles(f4[i]) ## visiting city quartiles feature wise
        l2_f3 += l2_norm(quart_f3_h, quart_f3_v) ##home city l2-norm feature wise
    for i in range(len(f5)):
        quart_f5_h = quartiles(f5[i]) ##home city quartiles feature wise
        quart_f5_v = quartiles(f6[i]) ## visiting city quartiles feature wise
        l2_f5 += l2_norm(quart_f5_h, quart_f5_v) ##home city l2-norm feature wise
    for i in range(len(f7)):
        quart_f7_h = quartiles(f7[i]) ##home city quartiles feature wise
        quart_f7_v = quartiles(f8[i]) ## visiting city quartiles feature wise
        l2_f7 += l2_norm(quart_f7_h, quart_f7_v) ##home city l2-norm feature wise
       
    return [l2_f1, l2_f3, l2_f5, l2_f7].sort()