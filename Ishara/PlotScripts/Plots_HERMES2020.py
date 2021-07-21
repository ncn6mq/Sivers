import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import Plot_Definitions *


# Here the data sets are listed and collected into and Array called "DataFilesArray"
#Dat1='Data/HERMES_p_2009.csv'
Dat2='Data/HERMES_p_2020.csv'
#Dat3='Data/COMPASS_d_2009.csv'
#Dat4='Data/COMPASS_p_2015.csv'
DataFilesArray=[Dat2]

result_Hermes_2020=([ 7.59045427e+00,  9.59480166e-01,  2.29080664e+00,  9.82637663e+00,
        2.05002362e-01, -4.71272523e+00,  4.81910776e-01,  5.67546672e-06,
        1.49041173e+00,  4.52841574e+00,  1.74496474e+00,  6.08183167e+00,
        8.69154831e+00])

result_Hermes_2020_err=array([ 7.59045427e-01,  9.59480166e-02,  2.29080664e-01,  9.82637663e-01,
        2.05002362e-02, -4.71272523e-01,  4.81910776e-02,  5.67546672e-07,
        1.49041173e-01,  4.52841574e-01,  1.74496474e-01,  6.08183167e-01,
        8.69154831e-01])

test_parm_1 = param_samples(DataFilesArray,result_Hermes_2020,result_Hermes_2020_err)

f1=plt.figure(1,figsize=(12,10))
plotSiversQBand(2,test_parm_1,'blue','$u$',result_Hermes_2020)
plotSiversQBand(1,test_parm_1,'red','$d$',result_Hermes_2020)
plotSiversQBand(3,test_parm_1,'green','$s$',result_Hermes_2020)
plt.legend(loc=1,fontsize=20,handlelength=3)
plt.title("HERMES 2020",fontsize=20)
plt.xlabel('$k_{\perp}$',fontsize=20)
plt.ylabel('$x \Delta^N f(x,k_{\perp})$',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(-0.01,0.006)
plt.legend(loc=1,fontsize=20,handlelength=3)
f1.savefig('HERMES2020QuarkSivers.pdf')


f2=plt.figure(2,figsize=(12,10))
plotSiversAntiQBand(-2,test_parm_1,'blue','$u_{sea}$',result_Hermes_2020)
plotSiversAntiQBand(-1,test_parm_1,'red','$d_{sea}$',result_Hermes_2020)
plotSiversAntiQBand(-3,test_parm_1,'green','$s_{sea}$',result_Hermes_2020)
plt.title("HERMES 2020",fontsize=20)
plt.xlabel('$k_{\perp}$',fontsize=20)
plt.ylabel('$x \Delta^N f(x,k_{\perp})$',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc=4,fontsize=20,handlelength=3)
f2.savefig('HERMES2020AntiQuarkSivers.pdf')