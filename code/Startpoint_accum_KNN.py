####################################读取数据集，画出最佳路径，保存为图片########################
import math, sys
import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import os
from scipy.misc import imread
import matplotlib.cbook as cbook
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from collections import Counter
 
class cycdat:
        def __init__(self, vx, vy, wind, pres, data):
                self.x = vx
                self.y = vy
                self.w = wind
                self.p = pres
                self.d = data   #此网格中有多少个数据点，因此可以对速度求和并求平均值
                #self.t = temp               
######################可以导入此文件夹的所有的TXT文件###################################
#f = open('./A/CH1949BST.txt', 'r+')
                # 定义figure
year=[]
month=[]
day=[]
I=[]
LAT=[]
LONG=[]
PRES=[]
WND=[]
matrix=np.full((2204,2204),0,dtype=int)

for filename in os.listdir("C:/Users/whuxu/Desktop/code/A"):  
#     print("C:/Users/whuxu/Desktop/code/A/" + filename)
    with open("C:/Users/whuxu/Desktop/code/A/"+filename) as f:
    #with pd.read_csv("C:/Users/whuxu/Desktop/code/A/"+filename)as f:
     #   X = np.array(f.ix[:, 0:4]) 	# end index is exclusive
      #  y = np.array(f['class']) 	# showing you two ways of indexing a pandas df
#        plt.figure()                         
        k = f.read() 
        lines = k.split('\n')     #用空格键进行分割
        flag=0;
        for l in lines:
            lastyear = None
            sp = l.split(' ')        #用空格进行分割，分成不同的数组元素
            sp = [y for y in sp if y != '']   
           
            try:
                    if (((sp[1] == '0')or(sp[1] == '1')or(sp[1] == '2')or(sp[1] == '3')or(sp[1] == '4')or(sp[1] == '5')or(sp[1] == '6')or(sp[1] == '9'))and(flag==0)): 
                        #date
                     #       print(l)
                            t1 = sp[0]
                            year1 = t1[:4]                            
                            month1 = int(t1[5:6])                            
                            day1 = t1[7:8]
                            lat1 = int(sp[2])/10                           
                            lon1 = int(sp[3])/10 
                            wind1 = int(sp[5])
                            pres1 = int(sp[4])
                            #intensity
                            if(int(sp[5])>=10.8):                                
                                    WND.append(wind1)                            
                                    year.append(year1)                            
                                    day.append(day1)                            
                                    month.append(month1)
                                    I.append(sp[1])
                                    PRES.append(sp[4])
                                    LAT.append(lat1)
                                    LONG.append(lon1)  
                                    flag=1;
                                    print((year1))
                                    print((month1))
                                    print((day1))
                                    print((sp[1]))
                                    print((sp[4]))
                                    print((lat1))
                                    print((lon1))
                                    
#                                    print(PRES)
#                            print(year)
                            #热带或外来的
                            if sp[1] == '6':
                                    ex1 = False
                            else:
                                    ex1 = True
                                                            #position

                           
                            if lastyear is not None and lastlon < 180.0 and lastlat < 50.0 and (t1[6:8] == "00" or t1[6:8] == "06" or t1[6:8] == "12" or t1[6:8] == "18"):
                                    dlat = lat1 - lastlat
                                    dlon = lon1 - lastlon
                                    dwind = wind1 - lastwind
                                    dpres = pres1 - lastpres
                                   
                                    grid[lastmonth - 1][math.floor(lastlat)][math.floor(lastlon) - 100].x += dlat
                                    grid[lastmonth - 1][math.floor(lastlat)][math.floor(lastlon) - 100].y += dlon
                                    grid[lastmonth - 1][math.floor(lastlat)][math.floor(lastlon) - 100].w += dwind
                                    grid[lastmonth - 1][math.floor(lastlat)][math.floor(lastlon) - 100].p += dpres
                                    grid[lastmonth - 1][math.floor(lastlat)][math.floor(lastlon) - 100].d += 1
                           
                            lastyear = year1
                            lastmonth = month1
                            lastday = day1
                            lastlat = lat1
                            lastlon = lon1
                            lastwind = wind1
                            lastpres = pres1
                    if sp[0] == '66666': #reset all of the last things and bypass adding a point between storm end & start positions                                                         
                            flag=0;
                            lastyear = None     

            except:
                  print("end of file... probably")
print(len(LONG))
print(len(LAT))
datafile = cbook.get_sample_data('C:/Users/whuxu/Desktop/code/China_map/China.png')
img = imread(datafile)                            
#                            plt.plot(LONG,LAT,zorder=0.5)
plt.scatter(LONG, LAT,s=0.1,c='r',zorder=0.5)
plt.imshow(img, zorder=0, extent=[70, 150, 0, 55])    
plt.show()


# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# ============================== KNN with k = 3 ===============================================
# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
acc = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the knn classifier for k = 3 is %d%%' % acc)
# ============================== parameter tuning =============================================
# creating odd list of K for KNN
myList = list(range(0,50))
neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('\nThe optimal number of neighbors is %d.' % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# =============================================================================================
#					Part II
# =============================================================================================
# ===================================== writing our own KNN ===================================

def train(X_train, y_train):
	# do nothing
	return

def predict(X_train, y_train, x_test, k):
	# create list for distances and targets
	distances = []
	targets = []

	for i in range(len(X_train)):
		# first we compute the euclidean distance
		distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
		# add it to list of distances
		distances.append([distance, i])

	# sort the list
	distances = sorted(distances)

	# make a list of the k neighbors' targets
	for i in range(k):
		index = distances[i][1]
		#print(y_train[index])
		targets.append(y_train[index])

	# return most common target
	return Counter(targets).most_common(1)[0][0]

def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
	# check if k is not larger than n
	if k > len(X_train):
		raise ValueError

	# train on the input data
	train(X_train, y_train)

	# predict for each testing observation
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test[i, :], k))
# ============================== testing our KNN =============================================
# making our predictions
predictions = []
try:
	kNearestNeighbor(X_train, y_train, X_test, predictions, 7)
	predictions = np.asarray(predictions)

	# evaluating accuracy
	accuracy = accuracy_score(y_test, predictions) * 100
	print('\nThe accuracy of OUR classifier is %d%%' % accuracy)

except ValueError:
	print('Can\'t have more neighbors than training samples!!')