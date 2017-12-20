import csv
import numpy as np

#sampl = np.random.uniform(0,1,size=(1,3))
#print float(sampl[0,1])*.1
#sampl = list(sampl)
#print sampl, len(sampl)
#dot_sampl = np.dot(4, 5)
#print dot_sampl

def requiredArg (str,num):
    print str,num

def myPerceptron (row):
    w = np.random.uniform(0,1,size=(1,64))
    bias = -1
    myWeightList=[]
    for i in range(64):
        #print i
        myWeightList.append(w[0,i])
    #print myWeightList,myWeightList[4]

    totalAttempt = 0
    #actual_class = 1
    while 1:
        rowtotal = 0
        col_count = 0
        ele_count = 0
        predicted_class = -1
        #print row[0:]
        for column in row[0:]:
            # ignore 1st row and last col for dot product calculation
            #print col_count,float(column)
            if col_count == 64:
                actual_class = float(column)
                #print col_count,actual_class
            if col_count < 64:
                ele_count = ele_count + 1
                #print col_count,float(column),float(w[0,col_count]),bias
                #rowtotal += float(column)*float(w[0,col_count])+bias
                rowtotal += float(column)*float(myWeightList[col_count])+bias
                #print(row[0], rowtotal, ele_count)
            col_count = col_count + 1
        if rowtotal > 0:
            predicted_class = 1
        if actual_class == predicted_class:
            #return totalAttempt,rowtotal,ele_count
            return totalAttempt
        else:
            weight_counter = 0;
            for column in row[-1:]:
                if weight_counter < 64:
                    #w[0,weight_counter] = float(w[0,weight_counter]) + float(column)*float(actual_class)
                    myWeightList[weight_counter] = float(myWeightList[weight_counter]) + float(column)*float(actual_class)
                weight_counter = weight_counter + 1
                print myWeightList
            totalAttempt = totalAttempt + 1
            #print totalAttempt


with open('/Users/lanqing/Desktop/machine/digits_binary.csv', 'rb') as csvfile:
    row_count = 0
    for row in csv.reader(csvfile):
        #if row_count > 0:
        if row_count == 1:
            #totalAttempt,rowtotal,ele_count = myPerceptron(row)
            totalAttempt = myPerceptron(row)
            print totalAttempt,1/totalAttempt
        #print row_count,totalAttempt,rowtotal,ele_count
        row_count = row_count + 1
        
