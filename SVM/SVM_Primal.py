# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 03:28:28 2020

@author: benny
"""

import csv
import collections
import math 
import copy
import numpy as np
import random

def PredictIsCorrect(w,x,y):
    predict_ans=0
    if x.dot(w) >0:
        predict_ans=1
    else:
        predict_ans=-1
        
    return y-predict_ans;


def SVM_helper(w,x,y):
    return (x.dot(w)*y) <=1
    
    
    



def SVM_primal_sgd(w,op):
    #shuffle data
    next_w=w
    shuffle_list=list(range(0, train_lines))
    random.shuffle(shuffle_list)
    for i in shuffle_list:
        scaler=C*learning_rate*result_list[i]
        temp_a=next_w*(1-learning_rate) 
        temp_b=(data_list[i].reshape(5,1))*scaler
        if SVM_helper(next_w,data_list[i],result_list[i]):
            next_w=np.add(temp_a,temp_b)
        else:
            next_w=(1-learning_rate)*next_w
    op.append(w)
    return next_w


number_features=4;
T=100;
learning_rate=0.5
C=700/873

data_list=[];
result_list=[];
train_lines=0
output=[]
test_list=[]
test_result_list=[]


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


inital_weight=np.zeros((number_features+1,1))

current_weight=inital_weight;
original_LR=learning_rate
for t in range(0,T):
    current_weight=SVM_primal_sgd(current_weight,output)
        #update learning rate
    learning_rate=original_LR/(1+t*original_LR/2)



#Read test data

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
final_weight=(output[len(output)-1])
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
final_weight=(output[len(output)-1])
error_counter=0
for i in range(0,len(train_list)):
    
    result=PredictIsCorrect(final_weight,train_list[i],train_result_list[i])
    if result !=0:
        error_counter=error_counter+1
    

error_train=error_counter/len(train_list)







print("The weight = ",final_weight.reshape(1,5),"\n"," the test_prediction error = ",\
      error_test,"\n"," the train_prediction error = ",error_train)









