import cv2
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.ndimage import rank_filter
import scipy.stats as st
import statsmodels.api as sm
from itertools import combinations
from gingerit.gingerit import GingerIt
import pysbd, re # pip install pysbd
import warnings

# TO find the best fit distribution of histogram
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)


# To get local maxima ans minima
def predictTrend(hgt,ylabels,xlabel,bar_type,x_title,y_title,  inter):
    trend_str = ''
    xlbls = xlabel
    local_max = argrelextrema(hgt, np.greater)
    local_min = argrelextrema(hgt, np.less)
    order = np.array([0] * len(xlbls))
    order[local_min] = -1
    order[local_max] = 1

    if  inter == False or np.count_nonzero(order)<2:
        # In the case of interclass trends
        if inter == True:
            neg_count = len([num for num in hgt if num <= 0])
            if neg_count>(len(hgt)-neg_count):
                hgt = np.array([num*-1 for num in hgt])
            trend_str=". The bar height differnce between "+ylabels[0]+" and "+ylabels[1]
            if y_title != '_':
                trend_str=". The "+y_title+" differnce between "+ylabels[0]+" and "+ylabels[1]
        elif ylabels =='Y':
            trend_str=". The Y axis value"
            if y_title != '_':
                trend_str=". The "+y_title
        elif ylabels =='freq':
            trend_str=". The frequency"
        else:
            trend_str=". The "+ylabels
            if y_title != '_':
                trend_str=". The "+y_title+" of "+ylabels

        if list(order)==[0]*len(xlbls):
            if(int(hgt[0])<int(hgt[1])):
                trend_str += " has an overall increasing trend"
                if( int(hgt[len(hgt)-2])>int(hgt[len(hgt)-1]) ):
                    trend_str += " till "+str(xlbls[len(xlbls)-2])+" and ends with a drop in "+str(xlbls[len(xlbls)-1])
                else:
                    trend_str += " from "+str(xlbls[0])+" to "+str(xlbls[len(xlbls)-1])
            elif(int(hgt[0])>int(hgt[1])):
                trend_str += " has an overall decreasing trend"
                if( int(hgt[len(hgt)-2])<int(hgt[len(hgt)-1]) ):
                    trend_str += " till "+str(xlbls[len(xlbls)-2])+" and ends with a peak in "+str(xlbls[len(xlbls)-1])
                else:
                    trend_str += " from "+str(xlbls[0])+" to "+str(xlbls[len(xlbls)-1])
            else:
                if(int(hgt[0])!=int(hgt[len(hgt)-1])):
                    trend_str += " is uniform with "+str(int(hgt[0]))+" till "+str(xlbls[len(xlbls)-2])+" and finally ends with "+str(int(hgt[len(hgt)-1]))+" in "+str(xlbls[len(xlbls)-1])
                else:
                    trend_str += " is uniform with "+str(int(hgt[0]))+" throughout the entire period"

        elif np.count_nonzero(order)<4:
            # speak about global maximum and min
            ht= hgt.tolist()
            xlbls2 = xlbls
            xlbls2[ht.index(max(ht))] = str(xlbls[ht.index(max(ht))]) + ' the maximum value'
            xlbls2[ht.index(min(ht))] = str(xlbls[ht.index(min(ht))]) + ' the minimum value'

            trend_str += " starts with "+str(int(hgt[0]))+" at "+x_title+' '+str(xlbls2[0])+" then "
            j=1
            while j<len(order):
                if order[j]==-1:
                    if list(order[:j])==[0]*j:
                        trend_str += "declines till "+str(xlbls2[j])
                    else:
                        trend_str += ", followed by a decreasing trend till "+str(xlbls2[j])
                elif order[j]==1:
                    if list(order[:j])==[0]*j:
                        trend_str += "increases till "+str(xlbls2[j])
                    else :
                        trend_str += ", followed by an increasing trend till "+str(xlbls2[j])
                j+=1
            if(order[j-2]!=0):
                trend_str += ", and finally ends with "+str(int(hgt[j-1]))+" in "+str(xlbls2[j-1])
            else :
                if(hgt[j-1]<hgt[j-2]):
                    trend_str += ", and ends with a decreasing trend till "+str(xlbls2[j-1])
                else:
                    trend_str += ", and ends with a decreasing trend till "+str(xlbls2[j-1])
            xlbls2[ht.index(max(ht))] = xlbls2[ht.index(max(ht))].replace(' the maximum value','')
            xlbls2[ht.index(min(ht))] = xlbls2[ht.index(min(ht))].replace(' the minimum value','')
        else:
            #Just discuss the maximum and minmium value
            ht= hgt.tolist()
            trend_str += ' has it maximum and minmum values '+str(int(max(ht)))+' and '+str(int(min(ht)))+' at '+str(xlbls[ht.index(max(ht))])+', and '+str(xlbls[ht.index(min(ht))])+" respectively"

    return trend_str

def simplebarsumm(yvals,ylabels,slabs,bar_type,x_title,y_title, inter):
    if len(slabs)>6 or inter == True:
        # Speak about maxima and minima
        Summ = predictTrend(yvals,ylabels,slabs,bar_type,x_title,y_title, inter)
    else :
        if x_title=='_':
            x_title = 'labels'
        if str(slabs[0]).isnumeric():
            Summ = '. For the '+str(x_title)+' ranging form '+str(slabs[0])+' - '+str(slabs[-1])+' at the interval '+str(abs(slabs[0]-slabs[1]))
            Summ += ', the '+str(y_title)
            if not (ylabels == 'Y' or  ylabels == 'freq'):
                Summ += " of "+ylabels
            Summ += ' are '
            for i in yvals[:-1]:
                Summ += str(round(i,2))+', '
            Summ += 'and '+str(round(yvals[-1],2))+' respectively'
        else :
            Summ = '. The '+str(y_title)
            if not (ylabels == 'Y' or  ylabels == 'freq'):
                Summ += " of "+ylabels
            Summ += ' are '
            for i in yvals[:-1]:
                Summ += str(round(i,2))+', '
            Summ += 'and '+str(round(yvals[-1],2))
            Summ += ' for the '+str(x_title)+" "
            for i in slabs[:-1]:
                Summ += str(i)+', '
            Summ += 'and '+str(slabs[-1])+' respectively'

    return Summ

segmentor = pysbd.Segmenter(language="en", clean=False)
subsegment_re = r'[^;:\n•]+[;,:\n•]?\s*'

def GrammerCorrect(par):
    fixed = []
    for sentence in segmentor.segment(par):
        if len(sentence) < 300:
            fixed.append(GingerIt().parse(sentence)['result'])
        else:
            subsegments = re.findall(subsegment_re, sentence)
            if len(subsegments) == 1 or any(len(v) < 300 for v in subsegments):
                # print(f'Skipped: {sentence}') // No grammar check possible
                fixed.append(sentence)
            else:
                res = []
                for s in subsegments:
                    res.append(GingerIt().parse(s)['result'])
                fixed.append("".join(res))
    return " ".join(fixed)

def genrateSumm_Bar(file):
    imgno = file.split('/')[-1].split(".")[0].split("_")[-1]
    path = os.path.dirname(file)+'/'
    df = pd.read_csv(file)
    xlabel = (df.loc[ : , list(df)[0]]).values
    xlabs = []
    for i in xlabel:
        if isinstance(i, np.float64):
            xlabs += [int(round(i))]
        else :
            xlabs += [i]
    x_title = df['x-title'][0]
    y_title = df['y-title'][0]
    title = df['title'][0]
    bar_type = df['bar_type'][0]
    ylabels = list(df)[1:len(list(df))-4]
    if bar_type == 'Histogram':
        ylabels = ylabels[:-1]
    data = (df.loc[ : , ylabels]).values




    ### Visual Summary
    #Speak about starting line with titles
    if bar_type == 'Histogram':
        Summ = 'The plot depicts a '+bar_type
        if title !='_':
            Summ += ' illustrating the frequency of \''+title+'\''
        elif x_title !='_':
            Summ += ' illustrating the frequency of \''+x_title+'\''
    else:
        Summ = 'The plot depicts a '+bar_type+' Graph'
        if title !='_':
            Summ += ' illustrating '+title
        if x_title != '_' and y_title != '_':
            Summ +='. The plot is between '+y_title+' on y-axis over '+x_title+' on the x-axis'
        elif y_title != '_':
            Summ +='. The plot is having '+y_title+' on y-axis'
        elif x_title != '_':
            Summ +='. The plot is having '+x_title+' on x-axis'
        # speaking about legend
        if bar_type != 'Vertical Simple Bar' and bar_type != 'Horizontal Simple Bar':
            Summ +=' for '
            for i in range(len(ylabels)-1):
                Summ += str(ylabels[i])+", "
            Summ += "and "+str(ylabels[i+1])
        # interchanging titles in case of horizontal charts
        if bar_type == 'Horizontal Simple Bar' or bar_type == 'Horizontal Grouped Bar' or bar_type == 'Horizontal Stacked Bar':
            temp = y_title
            y_title = x_title
            x_title = temp
    # intra class differrences
    for i in range(len(ylabels)):
        Summ += simplebarsumm(data[:,i],ylabels[i],xlabs,bar_type,x_title,y_title,False)
    Summ = Summ.replace("_ ", "")
    Summ = Summ.replace("\n", " ").replace('\r', '')
    # inter class differrences
    if 'Grouped' in bar_type or 'Stacked' in bar_type:
        Summ2 = ''
        for x,y in list(combinations(range(len(ylabels)), 2)):
            Summ2 += simplebarsumm(data[:,x]-data[:,y],[ylabels[x],ylabels[y]],xlabs,bar_type,x_title,y_title,True)
        # Cummulative description in stacked bar
        if 'Stacked' in bar_type :
            Summ2 += simplebarsumm(np.sum(data, axis=1),'all catogeries cummulatively',xlabs,bar_type,x_title,y_title, False)
        Summ2 = Summ2.replace("_ ", "")
        Summ2 = Summ2.replace("\n", " ").replace('\r', '')
        if len(Summ2)>2:
            Summ = Summ +".\n\t"+ Summ2[2:]

    ### Statistical Summary
    if bar_type == 'Histogram':
        min_w = round(min(list(df['bin_width'])),2)
        freq = []
        for i in np.array(df.ix[:,0:3]):
            freq+=[int(i[0])]*int(i[1])
        data = pd.Series(freq)
        mode_id = list(df['freq']).index(max(list(df['freq'])))
        Summ += "The frequency bins of histogram range from "
        best_fit_name, best_fit_params = best_fit_distribution(data, len(df), None)
        best_dist = getattr(st, best_fit_name)
        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])

        Summ += str(round(df['bin_center'][0]))+' to '+str(round(df['bin_center'][len(df)-1]))+' with '+str(min_w)+' bin width. The mode of a histogram is '+str(round(df['bin_center'][mode_id]))+' with a frequency of '+str(int(round(df['freq'][mode_id])))+'. The frequency distribution of histogram is the '+best_fit_name+' with following parameters '+param_str+'.'
    else:
        if 'Simple Bar' in  bar_type:
            dat, xlabs= zip(*sorted(zip(np.round(data[:,0].tolist(), decimals=2), xlabs), reverse=True))
            if y_title != '_':
                Summ += '. The overall mean and standard deviation values of '+y_title+' are '+str(round(sum(dat)/len(dat),2))+' and '+str(round(np.std(data[:,i]),2))+' respectively'
            else:
                Summ += '. The overall mean and standard deviation values are '+str(round(sum(dat)/len(dat),2))+' and '+str(round(np.std(data[:,i]),2))+' respectively'
        # For Catogeorical Graphs
        else:
            # To represent ranges of all groups
            if y_title != '_':
                Summ += '. The standard deviation values of '+y_title+' for catogeries \''
            else:
                Summ += '. The standard deviation values for catogeries \''
            for i in range(len(ylabels)-1):
                Summ += str(ylabels[i])+'\', '
            Summ += 'and \''+str(ylabels[-1])+'\' are '
            for i in range(len(ylabels)-1):
                Summ += str(round(np.std(data[:,i]),2))+', '
            Summ += 'and '+str(round(np.std(data[:,-1]),2))+' respectively '

            # Check for Correlation
            corr_mat = np.triu(df.iloc[:,1:len(list(df))-4].corr(method='spearman'), k=1)

            x,y=np.nonzero(abs(corr_mat)>0.6)
            # remove transtivity between items
            found_trnas = False
            test_dict = {}
            for i in set(x):
                test_dict[i] = [y[j] for j in range(len(x)) if i==x[j]]
            for i in set(y):
                if i not in test_dict:
                    test_dict[i] = [x[j] for j in range(len(y)) if i==y[j]]
                else :
                    test_dict[i] += [x[j] for j in range(len(y)) if i==y[j]]
            lst = [sorted([k]+v) for k, v in test_dict.items()]
            if len(lst)>1 and (lst.count(lst[0]) == len(lst)):
                found_trnas = True
                if len(x) == len([True for j in range(len(x)) if corr_mat[x[j],y[j]]>0]):
                    Summ += '. The categories \''
                    for i in range(len(lst[0])-1):
                        Summ += str(ylabels[lst[0][i]])+"\', "
                    Summ += "and \'"+str(ylabels[lst[0][-1]])+'\' are positively correlated with one another'
                elif len(x) == len([True for j in range(len(x)) if corr_mat[x[j],y[j]]<0]):
                    Summ += '. The categories \''
                    for i in range(len(lst[0])-1):
                        Summ += str(ylabels[lst[0][i]])+"\', "
                    Summ += "and \'"+str(ylabels[lst[0][-1]])+'\' are negatively correlated with one another'
                else:
                    found_trnas = False

            for j in range(len(x)):
                if not found_trnas:
                    if corr_mat[x[j],y[j]]>0:
                        Summ += '. The categories \''+str(ylabels[x[j]])+"\' and \'"+str(ylabels[y[j]])+'\' are positively correlated '
                    else:
                        Summ += '. The categories \''+str(ylabels[x[j]])+"\' and \'"+str(ylabels[y[j]])+'\' are negatively correlated '
                pos = np.count_nonzero((data[:,x[j]]-data[:,y[j]])>0)
                neg = np.count_nonzero((data[:,x[j]]-data[:,y[j]])<0)
                if y_title!= '_':
                    t = ' the '+y_title+' of'
                else:
                    t = ''
                if pos<neg and pos == 1:
                    k = np.nonzero((data[:,x[j]]-data[:,y[j]])>0)[0][0]
                    Summ += '. All except for '+str(xlabs[k])+t+' \''+str(ylabels[y[j]])+'\' is greater than \''+str(ylabels[x[j]])+'\''
                elif neg<pos and neg == 1:
                    k = np.nonzero((data[:,x[j]]-data[:,y[j]])<0)[0][0]
                    Summ += '. All except for '+str(xlabs[k])+t+' \''+str(ylabels[y[j]])+'\' is lesser than \'\''+str(ylabels[x[j]])+'\''
                elif(np.count_nonzero((data[:,x[j]]-data[:,y[j]])<0) == 1):
                    k = np.nonzero((data[:,x[j]]-data[:,y[j]])==0)[0][0]
                    Summ += '. All except for '+str(xlabs[k])+t+' \''+str(ylabels[y[j]])+'\' is equal to \''+str(ylabels[x[j]])+'\''


    Summ = GrammerCorrect(Summ+'.')
    text_file = open(path+"summ_"+str(imgno)+".txt", "w")
    n = text_file.write(Summ)
    text_file.close()


