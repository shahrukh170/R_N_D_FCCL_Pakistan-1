import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import math
from math import atan
from math import asin
from math import sqrt
from math import pow
import numpy as np
from pprint import pprint
from sklearn.metrics import mean_squared_error
import scipy.interpolate
import collections
import json
import csv
#import NozzleContour_FlowAnalysis_ShockwaveProps_toExcel_ML
import warnings
warnings.filterwarnings("ignore")
#Data Source 
#https://finance.yahoo.com/quote/AAPL/history/

#------------------------------------------------------------------------------
# DATA LOADER ARRAYS
#------------------------------------------------------------------------------
x = []
y = []
Q = []
R = []
theta = []
v_PM = []
mach = []
Pr_exit = []
T_exit = []
V_exit = []
A_ratio = []
exit_area = []
m_rate = []
thrust = []
#------------------------------------------------------------------------------
# FIXED DATA ARRAYS [ WITH PREDICTIONS ]
#------------------------------------------------------------------------------
x2 = []
y2 = []
Q2 = []
R2 = []
theta2 = []
v_PM2 = []
mach2 = []
Pr_exit2 = []
T_exit2 = []
V_exit2 = []
A_ratio2 = []
exit_area2 = []
m_rate2 = []
thrust2 = []

def get_data(filename):
    with open(filename,'r') as csvfile:
        csvFileReader =csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            x.append(float(row[0]))
            y.append(float(row[1]))
            Q.append(float(row[2]))
            R.append(float(row[3]))
            theta.append(float(row[4]))
            v_PM.append(float(row[5]))
            mach.append(float(row[6]))
            Pr_exit.append(float(row[7]))
            T_exit.append(float(row[8]))
            V_exit.append(float(row[9]))
            A_ratio.append(float(row[10]))
            exit_area.append(float(row[11]))
            m_rate.append(float(row[12]))
            thrust.append(float(row[13]))
                      
            
    return

def make_prediction(x,y,index):
    y = np.reshape(y,(len(y),1))
    x = np.reshape(x,(len(x),1))
    #print("ok")
    svr_lin  = SVR()
    #print(model)
    
    #svr_poly = SVR(kernel='poly',degree=2)
    
    #svr_rbf  = SVR(kernel='rbf',gamma='auto')
    svr_lin.fit(x,y)
    #model.fit(x,y)
    
    pred_y = svr_lin.predict(y)
    
    score=svr_lin.score(x,y)
    #print("F-Score:",score)

    mse =mean_squared_error(y, pred_y)
    #print("Mean Squared Error:",mse)

    rmse = math.sqrt(mse)
   #print("Root Mean Squared Error:", rmse)
   
    #svr_poly.fit(x,y)
    
    #svr_rbf.fit(x,y)
   
    #plt.scatter(range(len(x)),y,color='black',label='Date')
    #plt.plot(range(len(x)),svr_rbf.predict(y),color='red',label='RBF model')
    #plt.plot(range(len(x)),svr_lin.predict(y),color='green',label='Linear model')
    #plt.plot(range(len(x)),svr_poly.predict(y),color='blue',label='Polynomial model')
    #plt.xlabel('X[mm]')
    #plt.ylabel('Y')
    #plt.title('Support Vector Regression')
    #plt.legend()
    #plt.show()
    #return svr_rbf.predict(index)[0],svr_lin.predict(index)[0],svr_poly.predict(index)[0]
    return svr_lin.predict(index)[0]

get_data('./308_Length_Nozzle_Ideal_Values.csv')
print(len(x))
score =0
mse = 0
rmse= 0
finalX   =    []
i = 0
index = 0

with open('./308_Length_Nozzle_Corrected_Values.csv', 'w+', newline='') as outfile:
    f = csv.writer(outfile)
    # Write CSV Header, If you dont need that, remove this line
    f.writerow(["x" ,"y","Q","R","Theta","v_PM", "Mach", "Pr_exit" , "T_exit","V_exit", "A_Ratio","Exit_Area","M_rate", "Thrust"])

    for i in range(0,len(x),1):
        print(round(( i / len(x)) * 100 ,2))
        if abs(mach[i]) > 2 :
            #x2.append(x[i])
            finalX.append(x[i])
            index2 = np.array([x[i]]);
            index2 = index2.reshape(1,1)
            #prediction= make_prediction(x,y,index2)
            #y2.append(y[i])
            finalX.append(y[i])
           
            #fixing  Q , R with value of y
            #index2 = np.array([xi]);
            #index2 = index2.reshape(1,1)
            prediction= make_prediction(x,Q,index2)
            #Q2.append(prediction)
            finalX.append(prediction)
            

            #index2 = np.array([xi]);
            #index2 = index2.reshape(1,1)
            prediction= make_prediction(x,R,index2)
            #R2.append(prediction)
            finalX.append(prediction)

            #index2 = np.array([xi]);
            #index2 = index2.reshape(1,1)
            prediction= make_prediction(x,theta,index2)
            #theta2.append(prediction)
            finalX.append(prediction)

            #index2 = np.array([xi]);
            #index2 = index2.reshape(1,1)
            prediction= make_prediction(x,v_PM,index2)
            #v_PM2.append(prediction)
            finalX.append(prediction)

            index2 = np.array([x[i]]);
            index2 = index2.reshape(1,1)
            prediction= make_prediction(x,mach,index2)
            #mach2.append(prediction)
            finalX.append(prediction)
        
            #index2 = np.array([xi]);
            #index2 = index2.reshape(1,1)
            prediction= make_prediction(x,Pr_exit,index2)
            #Pr_exit2.append(prediction)
            finalX.append(prediction)

            #index2 = np.array([xi]);
            #index2 = index2.reshape(1,1)
            prediction= make_prediction(x,T_exit,index2)
            #T_exit2.append(prediction)
            finalX.append(prediction)

            #index2 = np.array([xi]);
            #index2 = index2.reshape(1,1)
            prediction= make_prediction(x,V_exit,index2)
            #V_exit2.append(prediction)
            finalX.append(prediction)

            #index2 = np.array([xi]);
            #index2 = index2.reshape(1,1)
            prediction= make_prediction(x,A_ratio,index2)
            #A_ratio2.append(prediction)
            finalX.append(prediction)

            #index2 = np.array([xi]);
            #index2 = index2.reshape(1,1)
            prediction= make_prediction(x,exit_area,index2)
            #exit_area2.append(prediction)
            finalX.append(prediction)

            #index2 = np.array([xi]);
            #index2 = index2.reshape(1,1)
            prediction= make_prediction(x,m_rate,index2)
            #m_rate2.append(prediction)
            finalX.append(prediction)
                
            #index2 = np.array([xi]);
            #index2 = index2.reshape(1,1)
            prediction= make_prediction(x,thrust,index2)
            #thrust2.append(prediction)
            finalX.append(prediction)

            '''
            finalX.append(x2[i])
            finalX.append(y2[i])
            finalX.append(Q2[i])
            finalX.append(R2[i])
            finalX.append(theta2[i])
            finalX.append(v_PM2[i])
            finalX.append(mach2[i])
            finalX.append(Pr_exit2[i])
            finalX.append(T_exit2[i])
            finalX.append(V_exit2[i])
            finalX.append(A_ratio2[i])
            finalX.append(exit_area2[i])
            finalX.append(m_rate2[i])
            finalX.append(thrust2[i])
            #f.writerow([[x2[i],y2[i],Q2[i],R2[i],theta2[i],v_PM2[i],Pr_exit2[i],T_exit2[i],V_exit2[i],A_ratio2[i],exit_area2[i],m_rate2[i],thrust2[i]]))
            '''
            f.writerow(finalX)
            finalX = []
        
              

            #print(prediction,"@[",xi,",",y[i],"]")
        else:
           '''
           x2.append(x[i]) 
           y2.append(y[i])
           Q2.append(Q[i])
           R2.append(R[i])
           theta2.append(theta[i])
           v_PM2.append(v_PM[i])
           mach2.append(mach[i])
           Pr_exit2.append(Pr_exit[i])
           T_exit2.append(T_exit[i])
           V_exit2.append(V_exit[i])
           A_ratio2.append(A_ratio[i])
           exit_area2.append(exit_area[i])
           m_rate2.append(m_rate[i])
           thrust2.append(thrust[i])
           '''
           print("k2-original")
           finalX.append(x[i])
           finalX.append(y[i])
           finalX.append(Q[i])
           finalX.append(R[i])
           finalX.append(theta[i])
           finalX.append(v_PM[i])
           finalX.append(mach[i])
           finalX.append(Pr_exit[i])
           finalX.append(T_exit[i])
           finalX.append(V_exit[i])
           finalX.append(A_ratio[i])
           finalX.append(exit_area[i])
           finalX.append(m_rate[i])
           finalX.append(thrust[i])
           
           #f.writerow([[x2[i],y2[i],Q2[i],R2[i],theta2[i],v_PM2[i],Pr_exit2[i],T_exit2[i],V_exit2[i],A_ratio2[i],exit_area2[i],m_rate2[i],thrust2[i]]))
           f.writerow(finalX)
           finalX = []
        
    #for i in range(0,len(x2),1):
        
