####################################读取数据集，画出最佳路径，保存为图片########################
import math, sys
import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import imread
import matplotlib.cbook as cbook
 
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

for filename in os.listdir("C:/Users/whuxu/Desktop/code/A"):  
#     print("C:/Users/whuxu/Desktop/code/A/" + filename)
    with open("C:/Users/whuxu/Desktop/code/A/"+filename) as f:

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
                            datafile = cbook.get_sample_data('C:/Users/whuxu/Desktop/code/China_map/China.png')
                            img = imread(datafile)                            
#                            plt.plot(LONG,LAT,zorder=0.5)
                            plt.scatter(LONG, LAT,s=5,c='r',zorder=0.5)
                            del LONG[:]
                            del LAT[:]
                            plt.imshow(img, zorder=0, extent=[70, 150, 0, 55])                              
                            flag=0;
                            lastyear = None     

            except:
                  #plt.show()
                  STR=str(filename);
                  plt.savefig('C:/Users/whuxu/Desktop/code/Startpoint/'+STR+'.png',dpi=500)
                  fig=plt.figure()
                  print("end of file... probably")