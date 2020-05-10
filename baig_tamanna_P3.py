# -*- coding: utf-8 -*-
"""
Created: Mon Oct 14 2019
Author: Tamanna Baig
Program: Logistic Regression

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def plot_initial_datafile(file):
    type1 = file[0:150]
    type0 = file[151:]
    plt.scatter(type1.body_len, type1.dorsal_fin, color='b', label='TigerFish1', alpha = 0.4)
    plt.scatter(type0.body_len, type0.dorsal_fin, color='r', label='TigerFish0', alpha = 0.4)
    plt.legend()
    plt.xlabel("Body length in cms")
    plt.ylabel("Dorsal fin length in cms")
    plt.title("Type of the Fish")
    plt.legend()
    plt.savefig('baig_initial_plot.png')
    plt.show()

#Randomize the data:
def data_randomize(data):
    return data.sample(frac=1, random_state = 6820).reset_index(drop=True)

#Normalize all the columns in the data:
def data_normalize(data, col_list = None):
    if col_list == None:
        data=(data-data.mean())/data.std()
    else:
        for col in col_list:
            data[col] = (data[col]-data[col].mean())/data[col].std()
    
    return(data)

#Split data into training set and test set:
def split_train_test(data):
    test_ratio = int(len(data) * 0.3)
    
    test_set = data.iloc[:test_ratio]
    train_set = data.iloc[test_ratio:]

    tr_add = open('baig_train_set.txt', 'a')
    tr_add.writelines(str(len(train_set))+'\n')
    tr_add.close()
    
    train_set.to_csv('baig_train_set.txt', sep = '\t' , index = False, header = False, mode = 'a')
    
    tes_add = open('baig_test_set.txt', 'a')
    tes_add.writelines(str(len(test_set))+'\n')
    tes_add.close()
    
    test_set.to_csv('baig_test_set.txt', sep = '\t' , index = False, header = False, mode = 'a')
    
    return train_set, test_set

def hypothesis(x1, x2, w_list):
    sigmoid = 0
    sigmoid = -(w_list[0] + (w_list[1] * x1) + (w_list[2] * x2))
#    print("*************************************")
#    print(w_list[0])
#    print(w_list[1],x1)
#    print(w_list[2],x2)
#    print(sigmoid)
    h = 1/(1+ math.exp(sigmoid))
    #print(h)
    return h

def make_predictions(data, weights):
    result = list()
    for i, row in data.iterrows():
        x1 = row[0]
        x2 = row[1]
        y = hypothesis(x1, x2, weights)
    
        result.append((y, row[2]))
    return result


#Calculate J values    
def calculate_cost(data, weight):
    m = len(data)
    cost_sum = 0
    for i, row in data.iterrows():
        x1 = row[0]
        x2 = row[1]
        y = row[2]   
        #cost_sum = cost_sum + ((hypothesis(x1, x2, weight) - y)**2)
        cost_sum = cost_sum + (-y * np.log(hypothesis(x1, x2, weight)) - (1 - y) * np.log(1 - hypothesis(x1, x2, weight)))

    return (cost_sum/m).round(6)

def get_z(wi, x1, x2):
    if wi == 0:
        z = 1
    elif wi == 1:
    	z = x1
    elif wi == 2:
    	z = x2 
    
    return z

def update_weight(data, w_list, alpha = 0.01):
    upd_w = [None] * len(w_list)
    for wi in range(len(w_list)):
        temp = 0
        old_w = w_list[wi]
#        print("old_w: ",old_w)
        for j, row in data.iterrows():
            x1 = row[0]
            x2 = row[1]
            y = row[2]
            
            z = get_z(wi, x1, x2)
            temp = temp + (hypothesis(x1, x2, w_list) - y ) * z
        
        new_w = old_w - alpha * temp
#        print("new_w: ",new_w.round(6))
        upd_w[wi] = new_w.round(6)
    
    return upd_w

def plot_final_J(iterations, cost_list):
    print("Number of iterations: ", len(iterations))
    plt.scatter(iterations, cost_list)
    plt.xlabel("Iterations")
    plt.ylabel("Updated Costs J")
    plt.title("Cost J over Iterations")
    plt.savefig("baig_plot_final_J.png")
    plt.show()
    
def logistic_regression(dataset, w_list, iterations = 50, alpha = 0.01):
    cost_list = list()
    iteration = list()
    
    for it in range(iterations):
        w_list = update_weight(dataset, w_list, alpha)
        cost = calculate_cost(dataset, w_list)
        cost_list.append(cost)
        iteration.append(it)
        
    df = pd.DataFrame({'Iteration_count': iteration, 'J_values': cost_list})

    df.to_csv('cost_over_iter.csv')
    
#    Plot of Final J Values
#    plot_final_J(iteration, cost_list)
        
    return w_list, cost_list[-1]

def get_col_data(data,col_names):
    means = {}
    stds = {}
    for col in col_names:
        means[col] = data[col].mean()
        stds[col] = data[col].std()
    return means, stds

def final_J_values(data, weights, final_J):
    pred_values = make_predictions(data, weights)
    with open('predictions.csv', 'w') as f:
        for item in pred_values:
            f.write(str(item))
            
#    mean_e, median_e = calculate_errors(pred_values)
#    print("len of data: ", len(data))

    for i in range(len(weights)):
        print("w"+str(i)+": ", weights[i].round(6))
    print("-----------------")
    print("J: ", final_J)
    print("-----------------")

def normalize_x(x, data_mean, data_std):
	xN = (x-data_mean)/data_std
	return xN

def get_user_inputs(cols_mean, cols_std, w_list):
    print("Enter Details of the fish in cms:")
    user_x1 = float(input("Body length in cms: "))
    user_x2 = float(input("Dorsal fin length in cms: "))
    
    if user_x1 == 0 and user_x2 == 0:
        print("\nYou entered 0 for both values!!\nTerminating Program......")
        return
    
    user_x1 = normalize_x(user_x1, cols_mean['body_len'], cols_std['body_len'])
    user_x2 = normalize_x(user_x2, cols_mean['dorsal_fin'], cols_std['dorsal_fin'])
    
    #print("\nFinal Weights: \n", w_list)
    y = hypothesis(user_x1, user_x2, w_list)
    
    if(y > 0.5):
        fish = 'TigerFish1'
    else:
        fish = 'TigerFish0'
        
    print("***************************************************************")
    print("The predicted type of fish is: ", fish)
    print("***************************************************************\n")
    get_user_inputs(cols_mean, cols_std, w_list)
 
def main():
    #*****************************Loading the data****************************#
    #Initial Values
    #file_name = str(input("Enter the Input file name:  \n(For example - FF03.txt)\n"))
    file_name = 'FF03.txt'
    #'test_file.txt'
    alpha = 0.01
    w_list = [-1, 0, 1]
    it = 400
    col_names = ["body_len", "dorsal_fin", "fish_type"]
    
    file = pd.read_csv(file_name, sep='\t', lineterminator='\n',skiprows = [0],
                    header = None, names = col_names)
    
    #To plot the Initial Data points from the input file
    #plot_initial_datafile(file)
    
    #Randomize the data
    rdm_data = data_randomize(file)
    #print(rdm_data.head())
    
    #Normalize the data
    norm_data = data_normalize(rdm_data, ['body_len', 'dorsal_fin'])
    #print(norm_data.head())
   
    # Split data into training and testing sets, save them as txts
    train_set, test_set = split_train_test(norm_data)
    
    print("The Algorithm has started computing........\n")    
    w_list, final_J = logistic_regression(train_set, w_list, it, alpha)
    
    #Calculating mean and std values for x1 and x2
    cols_mean, cols_std = get_col_data(file, ['body_len', 'dorsal_fin'])
    
    #To print the Final Values
    #final_J_values(test_set, w_list, final_J)
    
    #To calculate the J value on test set
#    test_w_list, test_final_J = logistic_regression(train_set, w_list, it, alpha)
#    print("Final J value on test set: ", test_final_J)
    

    #To predict the GPA of a student by taking input from the user
    print("Program ready to accept input values!")
    print("Enter 0 for both values to exit!\n")
    get_user_inputs(cols_mean, cols_std, w_list)

main()


