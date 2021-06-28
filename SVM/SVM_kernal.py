# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 09:39:52 2020

@author: benny
"""

import numpy as np
from scipy.optimize import minimize

def PredictIsCorrect(w,train_x,test_x,y,k):
    #k is lernal function
    
    #1st step maping the test_x to high dimention
    N = len(train_x)
    new_test_x=np.empty((1,N))
    for i in range(N):
        new_test_x[0,i]=k(test_x,train_x[i])
    
    
    
    predict_ans=0
    if new_test_x.dot(w) >0:
        predict_ans=1
    else:
        predict_ans=-1
        
    return y-predict_ans;


def Gram():
    return 0;

def gram(train_x, k):
    N = len(train_x)
    K = np.empty((N, N))
    for i in range(N):
        for j in range(N):
        #    print("xi=",X[i])
        #    print("xj=",X[j])
            
            K[i, j] = k(train_x[i], train_x[j])

    return K


def gaussian_kernel(x, y, c):
    return np.exp(-np.sum(np.square(x - y)) / c)






def Loss_fun(a):
    tw=Sum_axy(kernel_data_x,data_y,a)
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
    

r=5
C=500/873.0
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

kernel = lambda x, y: gaussian_kernel(x, y, r)



#mapping the data_x to higher dimension by kernel
kernel_data_x = gram(data_x, kernel)


#summation of  a_i*y_i=0
constraints = ({'type': 'eq', 'fun': lambda x: np.dot(x, t), 'jac': lambda x: t})





inital_weight=np.ones((train_lines,1))
# 0<=a_i<=C
bnds = [(0, C) for i in range(train_lines)]



print('Initial loss: ' + str(Loss_fun(inital_weight)))

res = minimize(Loss_fun, inital_weight, constraints=constraints,bounds=bnds, method='SLSQP', options={},tol=1e-6)


#print('Optimized loss: ' + str(res.fun))


weight_a = res.x  # optimal Lagrange multipliers


support_idx = np.where(0 < weight_a)[0]  # index of points with a>0; i.e. support vectors
margin_idx = np.where((0 < weight_a) & (weight_a < C))[0]  # index of support vectors that happen to lie on the margin, with 0<a<C
print('Total %d data points, %d support vectors' % (train_lines, len(support_idx)))


print()

weight_a=filter_a(weight_a)



final_weight=Sum_axy(kernel_data_x,data_y,weight_a)


final_weight=final_weight.reshape(872,1)

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


#run test data   w,train_x,test_x,y,k

error_counter=0
for i in range(0,len(test_list)):
    
    result=PredictIsCorrect(final_weight,data_x,test_list[i],test_result_list[i],kernel)
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
    
    result=PredictIsCorrect(final_weight,data_x,train_list[i],train_result_list[i],kernel)
    if result !=0:
        error_counter=error_counter+1
    

error_train=error_counter/len(train_list)







print("\n"," the test prediction error = ",\
      error_test,"\n"," the train prediction error = ",error_train)



with open("HW4result.txt", "a") as f:
     f.write("\n-------------------------------------\n")
     f.write("C=")
     f.write(str(C))
     f.write("\n")
     f.write("r=")
     f.write(str(r))
     f.write("\n")
     
     f.write("The test prediction error = ")
     f.write(str(error_test))
     f.write("\n")
     
     f.write("The train prediction error = ")
     f.write(str(error_train))
     f.write("\n")
     f.write(str(len(support_idx)))
     
     f.close()






