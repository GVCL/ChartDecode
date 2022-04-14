import pandas as pd
import numpy as np

_SQRT2 = np.sqrt(2)

for imgno in range(1,5):
    # Dy = [60000,10,14,750] # vGB
    # path = "Data_Extraction/Generated_data/vertical_grouped_bar/gb0"
    # # Dy = [100,3500,12,700] # hGB
    # # path = "Data_Extraction/Generated_data/horizontal_grouped_bar/h_gb0"
    # #
    # # Dy = [140,5000,350,550] # SB
    # # path = "Data_Extraction/Generated_data/vertical_stacked_bar/sb0"
    # # Dy = [4000,9,350,20] # hSB
    # # path = "Data_Extraction/Generated_data/horizontal_stacked_bar/h_sb0"
    #
    # data1 = pd.read_csv(path+str(imgno)+"/Sheet "+str(imgno)+"-Table 1.csv",  sep=",", index_col=False).to_numpy()
    # data2 = pd.read_csv(path+str(imgno)+"/data.csv",  sep=",", index_col=False).to_numpy()
    # data1 = data1[:,1:-3]
    # data2 = data2[:,1:-4]
    # dy = Dy[imgno-1]
    #
    # # dy = np.amax(data1)-np.amin(data1)
    # # print(dy)
    # TPc = 0
    # FPc = 0 #comission errors, wrongly predicting exsisting val
    # FNc = 0 # omission errors, missing data/prediction
    # m = []
    # for k in range(len(data1[2])):
    #     for i in range(len(data1)):
    #         if (abs(data1[i][k]-data2[i][k])/dy <= 0.02):
    #             if data1[i][k]!=0 and data2[i][k]==0:
    #                 FNc += 1
    #             else:
    #                 TPc += 1
    #         else :
    #             FPc +=1
    #     m += [np.abs(data1[i][k]-data2[i][k])/data1[i][k] for i in range(len(data1)) if data1[i][k] !=0 ]
    # MAPE = np.sum(m)/len(m)
    # nMAE = np.sum(np.abs(data1[:,1:len(data1[2])]-data2[:,1:len(data1[2])]))/np.sum(data1[:,1:len(data1[2])])
    # prec = 0
    # recall = 0
    # F1src = 0
    # if (TPc+FPc) != 0:
    #     prec = TPc/(TPc+FPc)
    # if (TPc+FNc) != 0:
    #     recall = TPc/(TPc+FNc)
    # if (prec+recall) != 0:
    #     F1src = 2*prec*recall/(prec+recall)
    #
    # print(prec,recall,F1src,MAPE,nMAE)
    #


# print("__________________________________________")

    # Dy = [120,10,40,325000] # hB
    # path = "Data_Extraction/Generated_data/horizontal_simple_bar/h_bc0"
    # data1 = pd.read_csv(path+str(imgno)+"/Sheet "+str(imgno)+"-Table 1.csv",  sep=",", index_col=False).to_numpy()
    # data2 = pd.read_csv(path+str(imgno)+"/data.csv",  sep=",", index_col=False).to_numpy()

    Dy = [120,100,800000,175] # B
    path = "Data_Extraction/Generated_data/vertical_simple_bar/bc0"
    data1 = pd.read_csv(path+str(imgno)+"/orig_data0"+str(imgno)+".csv",  sep=",", index_col=False).to_numpy()
    data2 = pd.read_csv(path+str(imgno)+"/data.csv",  sep=",", index_col=False).to_numpy()

    data1 = data1[:,1]
    data2 = data2[:,1]
    dy = Dy[imgno-1]
    TPc = 0
    FPc = 0 #comission errors, wrongly predicting exsisting val
    FNc = 0 # omission errors, missing data/prediction
    for i in range(len(data1)):
        if (abs(data1[i]-data2[i])/dy <= 0.02):
            if data1[i]!=0 and data2[i]==0:
                FNc += 1
            else:
                TPc += 1
        else :
            FPc +=1
    m = [np.abs(data1[i]-data2[i])/data1[i] for i in range(len(data1)) if data1[i] !=0 ]
    MAPE = np.sum(m)/len(m)
    nMAE = np.sum(np.abs(data1-data2))/np.sum(data1)

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
