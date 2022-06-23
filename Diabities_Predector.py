#########################################
# Required Python Packages
#########################################
import io
from tkinter.messagebox import NO
import joblib
import requests
import sys
from ast import While
import multiprocessing
from multiprocessing.sharedctypes import Value
import string
from sys import argv
import threading
import pandas as pd
import os
import schedule
import time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import message
from email import encoders
import pickle


########################################
# Declaring Gloabal Variables
########################################
best_model = None 
G_Accuracy = 0

##########################################
# Function name : Write_log
# Description :   Create the log file and write the script information
# Author : Abhishek V Dahiphale
# Date : 03/04/2022
#########################################
def Write_log(model,Accuracy):
    dirName = "LogFiles"
    if not os.path.isdir(dirName):
        os.mkdir(dirName)
    else:
        pass

    Log_file = "scriptInfo.log" 
    Logfile_path = os.path.join(dirName,Log_file)
    if not os.path.isfile(Logfile_path):
        fd = open(Logfile_path,'a')
    else:
        pass    
    design = "*"*80
    start_data = "Time : %s"%time.ctime()
    fd = open(Logfile_path,'a')
    fd.write(design+"\n")
    fd.write("\n"+start_data+"\n")
    fd.write("model : "+str(model)+" With Accuracy : "+str(Accuracy)+"\n")
    fd.close()
    return Logfile_path


#########################################
# Function name : VotingClassifier
# Description :  Train the data Using VotingClassifier() algoritham
# Output : Best Accuracy with model 
# Author : Abhishek V Dahiphale
# Date : 03/04/2022
#########################################
def VotingClassifier_algoritham():
    print("Voting Classifier")
    X_train = None
    X_test = None
    Y_train = None
    Y_test = None
    X_train, X_test, Y_train, Y_test = Data_Manipulating()
    Best_model = None
    Max_Accuracy = float(0)
    clf1 = KNeighborsClassifier()
    clf2 = DecisionTreeClassifier()
    clf3 = LogisticRegression()
    clf4 =  RandomForestClassifier(n_estimators=250)
    
    clf = VotingClassifier(estimators=[('knn',clf1),('dt',clf2),('lr',clf3),('rf',clf4)])
    Best_model = clf.fit(X_train,Y_train)
    test = Best_model.predict(X_test)
    Max_Accuracy = accuracy_score(Y_test,test)

    return Max_Accuracy,Best_model






#########################################
# Function name : RandomForest
# Description :  Train the data Using RandomForestClassifier() algoritham
# Output : Best Accuracy with model 
# Author : Abhishek V Dahiphale
# Date : 03/04/2022
#########################################
def RandomForest():
    #print("RandomForest")
    X_train = None
    X_test = None
    Y_train = None
    Y_test = None
    X_train, X_test, Y_train, Y_test = Data_Manipulating()
    Best_model = None
    Max_Accuracy = float(0)
    for n_estimators in range(50,100,5):
        clf = RandomForestClassifier(n_estimators=n_estimators)
        model = clf.fit(X_train,Y_train)
        test = model.predict(X_test)
        Acc = accuracy_score(Y_test,test)
        if Max_Accuracy < Acc:
            Max_Accuracy = Acc
            Best_model = model

    return Max_Accuracy,Best_model

#########################################
# Function name : Logistic_Regression
# Description :  Train the data Using LogisticRegression() algoritham
# Output : Best Accuracy with model 
# Author : Abhishek V Dahiphale
# Date : 03/04/2022
#########################################
def Logistic_Regression():
    #print("Logistic Regression")
    X_train = None
    X_test = None
    Y_train = None
    Y_test = None
    X_train, X_test, Y_train, Y_test = Data_Manipulating()
    Best_model = None
    Max_Accuracy = float(0)
    clf = LogisticRegression()
    Best_model = clf.fit(X_train,Y_train)
    test = Best_model.predict(X_test)
    Max_Accuracy = accuracy_score(Y_test,test)

    return Max_Accuracy,Best_model


#########################################
# Function name : DecisionTreeClassifier_algorithm
# Description :  Train the data Using DecisionTreeClassifier() algoritham
# Output : Best Accuracy with model 
# Author : Abhishek V Dahiphale
# Date : 03/04/2022
#########################################
def DecisionTreeClassifier_algorithm():
    #print("Decession tree classifier")
    X_train = None
    X_test = None
    Y_train = None
    Y_test = None
    X_train, X_test, Y_train, Y_test = Data_Manipulating()
    Best_model = None
    Max_Accuracy = float(0)
    for max_depth in range(3,31):
        clf = DecisionTreeClassifier(max_depth=max_depth) 
        model=clf.fit(X_train,Y_train)
        test = model.predict(X_test)
        Acc = accuracy_score(Y_test,test)
        if Max_Accuracy < Acc:
            Max_Accuracy = Acc
            Best_model = model

    return Max_Accuracy,Best_model   



#########################################
# Function name : K_nearest_neighbour_algorithm
# Description :  Train the data Using KNeighborsClassifier() algoritham
# Output : Best Accuracy with model 
# Author : Abhishek V Dahiphale
# Date : 03/04/2022
#########################################
def K_nearest_neighbour_algorithm():
    #print("In the KNN")
    X_train = None
    X_test = None
    Y_train = None
    Y_test = None
    X_train, X_test, Y_train, Y_test = Data_Manipulating()
    Best_model = None
    Max_Accuracy = 0
    for n_neighbors in range(3,10):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        model=clf.fit(X_train,Y_train)
        test = model.predict(X_test)
        Acc = accuracy_score(Y_test,test)
        if Max_Accuracy < Acc:
            Max_Accuracy = Acc
            Best_model = model

    return Max_Accuracy,Best_model       

#########################################
# Function name : Data_Manipulating
# Description :  Function will ready the data for the Training
# Input : Path Of CSV File  
# Author : Abhishek V Dahiphale
# Date : 03/04/2022
#########################################
def Data_Manipulating():
    X_train = None
    X_test = None
    Y_train = None
    Y_test = None
    """url = "https://github.com/AbhishekDahiphale/CSV_File/blob/main/diabetes.csv"
    s=requests.get(url).content
    df=pd.read_csv(io.StringIO(s.decode('utf-8')),on_bad_lines='skip')"""
    Path = "diabetes.csv"
    if not os.path.isabs(Path):
       Path = os.path.abspath(Path)
    #responce = requests.get(url).content
    df = pd.read_csv(Path)

    #df = pd.read_csv('diabetes.csv',error_bad_lines=False)
   
    Dependent = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']]
    Independent = df[['Outcome']]

    X_train, X_test, Y_train, Y_test = train_test_split(Dependent,Independent,test_size=0.3) 
    
    return X_train, X_test, Y_train, Y_test



flag = True
#########################################
# Function name : Train_And_Test_data
# Description :  Train the model using different algoritham
# Input : X_train, X_test, Y_train, Y_test
# Output : Best Model 
# Author : Abhishek V Dahiphale
# Date : 03/04/2022
#########################################
def Train_And_Test_Data():
    Greater_Accuracy = 0
    global G_Accuracy
    global best_model
    algoritham = [K_nearest_neighbour_algorithm(),DecisionTreeClassifier_algorithm(),Logistic_Regression(),RandomForest(),VotingClassifier_algoritham()] 
    
    Accuracy_List = list()
    for i in range(len(algoritham)):
        Acc,model = algoritham[i]
        Accuracy_List.append([Acc,model])
    No = 0

    Greater_Accuracy = Accuracy_List[0][0]
    for i in range(1,len(Accuracy_List)):
        if Greater_Accuracy < Accuracy_List[i][0]:
            Greater_Accuracy = Accuracy_List[i][0]
            No = i

    Best_Model = Accuracy_List[No][1]
    if((G_Accuracy*100)<(Greater_Accuracy*100)):
        best_model = Accuracy_List[No][1]
        G_Accuracy = float(Greater_Accuracy)
        joblib.dump(best_model,'Model.pkl')
        print("In the if model is : "+str(best_model)+" and accuracy is : "+str(G_Accuracy))
        Write_log(best_model,G_Accuracy)
    
    print(str(Best_Model)+" : "+str(Greater_Accuracy*100))

#########################################
# Function name : check_input
# Description :  function is user for checking Diabities or Not using model which is in the pickle file
# Author : Abhishek V Dahiphale
# Date : 03/04/2022
#########################################
def check_input(data):
    model = None
    model = joblib.load('model.pkl')
    print(model)
    op=model.predict(data)
    print(data)
    print(op[0])
    return op[0]  


#########################################
# Function name : main
# Description :  Main function from where execution starts
# Author : Abhishek V Dahiphale
# Date : 03/04/2022
#########################################
def main():
    print("-------Diabities Predictor----")
    G_Accuracy = 0.00
    print("Application Name : "+argv[0])
    try:  
        schedule.every(5).seconds.do(Train_And_Test_Data) 
        while True:
            schedule.run_pending()
            time.sleep(1)
    except Exception as e : 
            print("Exception Occur : ",e)

             


#########################################
# Application starter
#########################################
if __name__ == "__main__":
    main()
   

    