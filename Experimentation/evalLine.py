import pandas as pd
import numpy as np

_SQRT2 = np.sqrt(2)

for imgno in range(1,16):
    data1 = pd.read_csv("SYNTHETIC_DATA/LINE/orig_sline"+str(imgno)+".csv",  sep=",", index_col=False).to_numpy()
    data2 = pd.read_csv("SYNTHETIC_DATA/LINE_RESULTS/data_sline"+str(imgno)+".csv",  sep=",", index_col=False).to_numpy()

    Dx = [90,8,14,10,14,11,12,11,14,12,6,11,11,11,11]
    Dy = [6.5,67,27,373816,1000,60,100,500,100,80,40,300,1500,20,3000]
    dx, dy = Dx[imgno-1], Dy[imgno-1]

    TPc = 0
    FPc = 0
    FNc = 0
    m = []

    for k in range(len(data1[0])-1):
        ids = list(range(len(data2)))
        t = 0
        for i in range(len(data1)):
            flag = True
            Cflag = True
            for j in ids:
                (a,b,c,d)=(data1[i][0],data1[i][k+1],data2[j][2*k],data2[j][2*k+1])
                if not np.isnan((a,b,c,d)).any():
                    if (abs(a-c)/dx <= 0.02):
                        M += [[b,d]]
                        Cflag = False
                        if (abs(b-d)/dy <= 0.02):
                            flag = False
                            ids.remove(j)
                            TPc += 1
                            break
            if flag:
                if Cflag:
                    FNc += 1
                else:
                    FPc += 1

    for k in range(1,len(data1[2])):
        ids = list(range(len(data1)))
        t = 0
        for i in range(len(data1)):
            flag = True
            Cflag = True
            for j in ids:
                 if (abs(data1[i][0]-data2[j][0])/dx <= 0.02):
                        Cflag = False
                        if (abs(data1[i][k]-data2[j][k])/dy <= 0.02):
                            flag = False
                            ids.remove(j)
                            TPc += 1
                            break
            if flag:
                if Cflag:
                    FNc += 1
                else:
                    FPc += 1
        m += [np.abs(data1[i][k]-data2[i][k])/data1[i][k] for i in range(len(data1))]
    MAPE = np.sum(m)/len(m)
    nMAE = np.sum(np.abs(data1[:,1:len(data1[2])]-data2[:,1:len(data1[2])]))/np.sum(data1[:,1:len(data1[2])])

    prec = 0
    recall = 0
    F1src = 0
    if (TPc+FPc) != 0:
        prec = TPc/(TPc+FPc)
    if (TPc+FNc) != 0:
        recall = TPc/(TPc+FNc)
    if (prec+recall) != 0:
        F1src = 2*prec*recall/(prec+recall)

    print(prec,recall,F1src,MAPE,nMAE)
