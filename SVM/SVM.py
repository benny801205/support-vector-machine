# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 09:39:52 2020

@author: benny
"""

import numpy as np
from scipy.optimize import minimize

def PredictIsCorrect(w,x,y):
    predict_ans=0
    if x.dot(w) >0:
        predict_ans=1
    else:
        predict_ans=-1
        
    return y-predict_ans;


def Loss_fun(a):

    tw=Sum_axy(data_x,data_y,a)
    return 0.5 * np.dot(tw, tw.T)-a.sum()

def Sum_axy(x,y,a):
    a=a.reshape(train_lines,1)
    N=x.shape[1]
    w=np.zeros(N)
    a1=np.diag(a[:,0])

    ay=np.dot(a1,y).reshape(train_lines,1)
    ay_eye=np.diag(ay[:,0])

    axy=np.dot(ay_eye,x)
    
    for i in range(0,train_lines):
        tt=axy[i,:].reshape(1,N)
        w=np.add(w,tt)
    
    return w


def filter_a(a):
    a=a.reshape(train_lines,1)
    for i in range(0,train_lines):
        temp=a.item((i,0))
        if temp >C:
            a.itemset((i, 0), C)
        elif temp <0:
            a.itemset((i, 0), 0)
        else:
            a.itemset((i, 0), temp)


    return a

########################
    
C=100/873
number_features=4;
data_list=[];
result_list=[];
train_lines=0

with open ( r"bank-note\train.csv" , 'r' ) as f :
    for line in f :
            train_lines=train_lines+1
            terms = line.strip().split(',' )
            
            s=['1']#add bias
            s.extend(terms)
            s=np.array(s[0:number_features+1]).astype(float)
            s=s.reshape(1,number_features+1)
            data_list.append(s)
            
            #switch 0 ot -1
            if int(terms[number_features])==0:
                result_list.append(-1);
            else:
                result_list.append(1)
            
            
    
f.close();


data_x=np.concatenate(data_list)
data_y=np.array(result_list)
#data_y=data_y.reshape(train_lines,1)




c1 = np.where(np.prod(data_x, axis=1) > 0)
t = np.ones(train_lines)  
t[c1] = -1  






#summation  a_i*y_i=0
constraints = ({'type': 'eq', 'fun': lambda x: np.dot(x, t), 'jac': lambda x: t})





inital_weight=np.ones((train_lines,1))
# 0<=a_i<=C
bnds = [(0, C) for i in range(train_lines)]



print('Initial loss: ' + str(Loss_fun(inital_weight)))

res = minimize(Loss_fun, inital_weight, constraints=constraints,bounds=bnds, method='SLSQP', options={},tol=1e-6)


print('Optimized loss: ' + str(res.fun))


weight_a = res.x  # optimal Lagrange multipliers


print()

weight_a=filter_a(weight_a)



final_weight=Sum_axy(data_x,data_y,weight_a)
print(final_weight)


final_weight=final_weight.reshape(5,1)

#Read test data
test_list=[]
test_result_list=[]
with open ( r"bank-note\test.csv" , 'r' ) as f :
    for line in f :
            terms = line.strip().split(',' )      
            s=['1']#add bias
            s.extend(terms)
            s=np.array(s[0:number_features+1]).astype(float)
            s=s.reshape(1,number_features+1)
            test_list.append(s)
            
            #switch 0 ot -1
            if int(terms[number_features])==0:
                test_result_list.append(-1);
            else:
                test_result_list.append(1) 
f.close();


#run test data

error_counter=0
for i in range(0,len(test_list)):
    
    result=PredictIsCorrect(final_weight,test_list[i],test_result_list[i])
    if result !=0:
        error_counter=error_counter+1
    

error_test=error_counter/len(test_list)



train_list=[]
train_result_list=[]

with open ( r"bank-note\train.csv" , 'r' ) as f :
    for line in f :
            terms = line.strip().split(',' )      
            s=['1']#add bias
            s.extend(terms)
            s=np.array(s[0:number_features+1]).astype(float)
            s=s.reshape(1,number_features+1)
            train_list.append(s)
            
            #switch 0 ot -1
            if int(terms[number_features])==0:
                train_result_list.append(-1);
            else:
                train_result_list.append(1) 
f.close();


#run test data
error_counter=0
for i in range(0,len(train_list)):
    
    result=PredictIsCorrect(final_weight,train_list[i],train_result_list[i])
    if result !=0:
        error_counter=error_counter+1
    

error_train=error_counter/len(train_list)







print("The weight = ",final_weight.reshape(1,5),"\n"," the test_prediction error = ",\
      error_test,"\n"," the train_prediction error = ",error_train)










