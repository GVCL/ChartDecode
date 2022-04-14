import os
import csv
import numpy as np

import Summary_Generation.PieLineDot
from Extract_DataTables.utils import *
from Summary_Generation.PieLineDot import genrateSumm_PLD

def extLine(filename):
    image_name = os.path.basename(filename).split(".png")[0]
    path = os.path.dirname(filename)+'/'
    graph_img = cv2.imread(filename)
    if(graph_img.shape[2]==4):
        graph_img[graph_img[:,:,3]==0] = [255,255,255,255]
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2RGB)
    kernel = np.ones((3,3), np.uint8)
    ht,wt,_ = graph_img.shape

    '''Canvas and Legend Extraction'''
    chart_dict, IS_MULTI_CHART, legend_colors, legend_names, bg_color, canvas_img = extractCanvaLeg(graph_img,'line')
    print("Canvas & Legend Extracted Sucessfully .. !")

    '''Components extraction'''
    title, y_title, ybox_centers, Ylabel, x_title, xbox_centers, Xlabel = extractLablTitl(graph_img,chart_dict,IS_MULTI_CHART)
    print("Chart Labels & Titles Extracted Sucessfully .. !")
    # segchart = viewSegChart(graph_img,chart_dict)
    # cv2.imwrite(path+"CCD_"+str(image_name)+".png",segchart)

    '''Chart Reconstruction'''
    img = graph_img.copy()
    if  IS_MULTI_CHART:
        d = chart_dict['legend']
        img = cv2.rectangle(img, (d['x'],d['y']), (d['w']+d['x'],d['h']+d['y']), (255, 255, 255), -1)

    Y_val = []
    d = chart_dict['canvas']

    '''NORMALIZE ALL THE HEIGHTS OF PIX VALUES OBTAINED'''
    t = np.array([[int(Ylabel[i])]+ybox_centers[i].tolist() for i in range(len(Ylabel)) if Ylabel[i].isnumeric()])
    if len(t)>=2:
        Ylabel = list(t[:,0])
        ybox_centers = t[:,1:]
    # To deal with duplicate values and make one as negative
    for i in np.unique(Ylabel):
        id=[j for j, val in enumerate(Ylabel) if i==val]
        if(len(id)==2):
            if(ybox_centers[id[0]][1]<ybox_centers[id[1]][1]):
                Ylabel[id[1]]*=-1
                neg_ids=np.where(ybox_centers[:,1] > ybox_centers[id[1]][1])[0]
            else:
                Ylabel[id[0]]*=-1
                neg_ids=np.where(ybox_centers[:,1] > ybox_centers[id[0]][1])[0]
            for i in neg_ids:
                Ylabel[i]*=-1
    t = np.array(sorted(np.concatenate((ybox_centers, np.array([Ylabel]).T), axis=1), key=lambda x: x[1], reverse= True))
    ybox_centers,Ylabel = (t[:,0:2],list(t[:,2]))
    normalize_scaley = (Ylabel[0]-Ylabel[1])/(ybox_centers[0][1]-ybox_centers[1][1])


    xbox_centers = xbox_centers[:,0]
    t = np.array([[int(Xlabel[i]), xbox_centers[i]] for i in range(len(Xlabel)) if Xlabel[i].isnumeric()])
    if len(t)>=2:
        Xlabel = list(t[:,0])
        xbox_centers = t[:,1]
    if isinstance(Xlabel[0],int) or isinstance(Xlabel[0],float):
        # If there are duplicate values change the neg values
        for i in np.unique(Xlabel):
            id=[j for j, val in enumerate(Xlabel) if i==val]
            if(len(id)==2):
                if(xbox_centers[id[0]]>xbox_centers[id[1]]):
                    Xlabel[id[1]]*=-1
                    neg_ids=np.where(xbox_centers < xbox_centers[id[1]])[0]
                else:
                    Xlabel[id[0]]*=-1
                    neg_ids=np.where(xbox_centers < xbox_centers[id[0]])[0]
                for i in neg_ids:
                    Xlabel[i]*=-1
        xbox_centers, Xlabel = zip(*sorted(zip(xbox_centers, Xlabel)) )
        xbox_centers, Xlabel = (list(xbox_centers),list(Xlabel))
        normalize_scalex = (Xlabel[0]-Xlabel[1])/(xbox_centers[0]-xbox_centers[1])
        defa_X_intrvl = (xbox_centers[1]-xbox_centers[0])
        Y_id = []
        Y_val = []
        for j in range(len(legend_colors)):
            mas = np.all(img==legend_colors[j],axis=2).astype(np.uint8)*255
            mas = cv2.dilate(mas, kernel, iterations=3)
            if(len(Y_val)==0):
                Y = np.array([[i,int(np.mean(np.argwhere(mas[:,i]!=0)))] for i in range(wt) if len(np.argwhere(mas[:,i]==255))!=0])
                Y_id = list(Y[:,0])
                Y_val += [list(Y[:,1])]
            else :
                Y = np.array([[i,int(np.mean(np.argwhere(mas[:,i]!=0)))] for i in Y_id if len(np.argwhere(mas[:,i]==255))!=0])
                Y_val = np.delete(np.array(Y_val),[Y_id.index(i) for i in (set(Y_id)-set(Y[:,0]))],axis=1).tolist()
                Y_id = list(Y[:,0])
                Y_val += [list(Y[:,1])]

        Y_val = np.around((((np.array(Y_val) - ybox_centers[0][1]) * normalize_scaley) + Ylabel[0]),decimals=1)

        flwchnz_ids = np.unique(sum([flwchnz(Y_val[i]) for i in range(len(Y_val))], []))

        kl = 0
        data = []
        inter_dat = []
        while kl < len(Y_id)-1:
            for ku in range(kl,len(Y_id)):
                if Y_id[ku] >= Y_id[kl]+defa_X_intrvl:
                    break
            if len(flwchnz_ids)!=0:
                fc_ids = np.unique(sum([flwchnz(Y_val[i,kl:ku]) for i in range(len(Y_val))], []))
                if fc_ids != []:
                    inter_dat+=[[Y_id[i]]+list(Y_val[:,i]) for i in fc_ids]
            data += [[Y_id[kl]]+list(Y_val[:,kl])]
            kl = ku
        if kl<len(Y_id):
            data += [[Y_id[-1]]+list(Y_val[:,-1])]
        # print("---\n",data, inter_dat, defa_X_intrvl)
        while len(flwchnz_ids)!=0 and  len(inter_dat)>=len(data):
            kl = 0
            defa_X_intrvl = defa_X_intrvl*0.5#*(len(data))/len(flwchnz_ids))
            data = []
            inter_dat = []
            while kl < len(Y_id)-2:
                for ku in range(kl,len(Y_id)):
                    if Y_id[ku] >= Y_id[kl]+defa_X_intrvl:
                        break
                fc_ids = np.unique(sum([flwchnz(Y_val[i,kl:ku]) for i in range(len(Y_val))], []))
                if fc_ids != []:
                    inter_dat+=[[Y_id[i]]+list(Y_val[:,i]) for i in fc_ids]
                data += [[Y_id[kl]]+list(Y_val[:,kl])]
                kl = ku
            if kl<len(Y_id):
                data += [[Y_id[-1]]+list(Y_val[:,-1])]
            print("Default interval used for reconstruction : ", defa_X_intrvl)

        '''NORMALIZE ALL THE PIX X VALUES OBTAINED'''
        data = np.array(data)
        Xlabel = np.round(Xlabel[0]-(xbox_centers[0] - data[:,0] +2)*normalize_scalex,decimals=1)
        data  = data[:,1:]
        l = []
        for j in np.unique(Xlabel):
            l += [i for i, v in enumerate(Xlabel) if v == j][1:]
        Xlabel = np.delete(Xlabel,l)
        data = np.delete(np.array(data),l,axis=0)
    else :
        data = []
        xbox_centers, Xlabel = zip( *sorted(zip(xbox_centers, Xlabel)) )
        # print(xbox_centers.astype(int))
        for j in range(len(legend_colors)):
            mas = np.all(img==legend_colors[j],axis=2).astype(np.uint8)*255
            mas = cv2.dilate(mas, kernel, iterations=3)
            Y = [0]*len(Xlabel)
            for id,i in enumerate(np.array(xbox_centers).astype(int)):
                if len(np.argwhere(mas[:,i]==255))!=0:
                    Y[id] = int(np.mean(np.argwhere(mas[:,i]!=0)))
            data += [Y]
        data = np.around((((np.array(data).T - ybox_centers[0][1]) * normalize_scaley) + Ylabel[0]),decimals=1)



    fig=plt.figure(figsize=(8,6.4))
    for i in range(len(data[0])):
        plt.plot(Xlabel, data[:,i], color=(np.array(legend_colors[i][::-1])/255))
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    if IS_MULTI_CHART:
        plt.legend(legend_names)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path+"Reconstructed_"+str(image_name)+".png")

    # Writing data to CSV file
    L = [['X']+legend_names+['chart_type','title','x-title','y-title']]
    if IS_MULTI_CHART:
        L = L + [[Xlabel[0]]+data[0].tolist()+['Line',title, x_title, y_title]]
    else:
        L = L + [[Xlabel[0]]+data[0].tolist()+['Simple Line',title, x_title, y_title]]

    L = L + [[Xlabel[i]]+data[i].tolist() for i in range(1,len(Xlabel))]
    with open(path+'data_'+str(image_name)+'.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(L)
    print("Chart Reconstruction Done .. !")

    genrateSumm_PLD(path+'data_'+str(image_name)+'.csv')
    print("Chart Summary Generated .. !")




