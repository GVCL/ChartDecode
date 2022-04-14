# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import colorcet as cc
#
# df = pd.read_csv("Syntheticdata/orig_sline1.csv")
# xlabels = (df.loc[ : , list(df)[0]]).values
# ylabels = list(df)[1:len(list(df))]
# print(xlabels,ylabels)
# data = (df.loc[ : , ylabels]).values
#
# for i in range(len(data[0])):
#     plt.plot(xlabels, data[:,i])
# plt.xlabel('year')
# plt.ylabel('Population in thousands')
# plt.title('Bear population in emerald forest')
# # plt.legend(ylabels)
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig("Syntheticdata/sline1.png")
# plt.show()

import matplotlib.pyplot as plt

sizes = [60, 80, 90, 55, 10, 30]

labels = ['US', 'UK', 'India', 'Germany', 'Australia', 'South Korea']

colors = ['lightskyblue', 'red', 'purple', 'green', 'gold']
plt.pie(sizes,labels=labels, startangle=90)#, colors=colors)
plt.legend(labels = labels, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.title("Programming Language Usage Among Developers")
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Syntheticdata/spie3.png")
