import numpy as np
import csv
import sys
import os
import copy
csvfile=open('ml-100k/u.data')
csvreader = csv.reader(csvfile, delimiter='\t')
data=[]
for row in csvreader:
    data.append(row)
csvfile.close()
testid=np.random.randint(1,100000, 10000)
testset=[]
k=1
for i in testid:
    i-=k
    k+=1
    testset.append(data[i])
print('Select test 10000 cases.')
mdata=np.array(data,dtype=int)
usernum=int(np.max(mdata[:,0]))
itemnum=int(np.max(mdata[:,1]))
print('Total user number is : '+str(usernum))
print('Total movie number is : '+str(itemnum))
fdata = np.array(np.zeros((usernum, itemnum)))
print('Formating matrix.')
for row in mdata:
    fdata[row[0]-1, row[1]-1] = row[2]
for case in testset:
    fdata[int(case[0])-1,int(case[1])-1]=0

def findKNNitem(indata, item):
    iid = int(item[1])
    uid = int(item[0])
    temp = copy.deepcopy(indata[:, iid-1])
    for j in range(itemnum):
        indata[:, j] -= temp
    indata = indata**3
    sumd=indata.sum(axis=0)
    max=sumd.max()
    nn=[]
    for l in range(5):
        while fdata[uid-1,sumd.argmin()] == 0:
            sumd[sumd.argmin()]=max
        nn.append(sumd.argmin())
    ratelist = []
    for j in range(5):
        ratelist.append(fdata[uid-1, nn[j]])
    rate = np.average(ratelist)
    error = np.absolute(int(item[2])-rate)
    return error


def findKNNuser(indata,item):
    iid = int(item[1])
    uid = int(item[0])
    temp = copy.deepcopy(indata[uid-1, :])
    for i in range(usernum):
        indata[i, :] -= temp
    indata = indata**3
    sumd=indata.sum(axis=1)
    max=sumd.max()
    nn=[]
    z=5
    for i in range(z):
        while fdata[sumd.argmin(),iid-1]==0:
            sumd[sumd.argmin()]=max
            if sumd.max()==sumd.min():
                z=i
                break

        nn.append(sumd.argmin())
    ratelist=[]
    for i in range(z):
        ratelist.append(fdata[nn[i], iid-1])
    rate = np.average(ratelist)
    error=np.absolute(int(item[2])-rate)
    return error

errorlist=[]
print('start test '+str(len(testset)))
n=1
for testcase in testset:
    if n % 100==0:
        print('Test has finished : '+str(n/100)+'%' )
    error1=findKNNuser(copy.deepcopy(fdata), testcase)
    error2=findKNNitem(copy.deepcopy(fdata), testcase)
    errorlist.append((error2/2+error1/2)**2)
    n+=1
print('test finished')
meanserror=np.average(errorlist)
print(meanserror)








