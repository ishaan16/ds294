import json
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def koren_plot(algoStats,algoStats_base):
    hist1,bins=np.histogram(algoStats['GMF'],bins=1000,range=(1,1000),density = True)
    hist1=np.cumsum(hist1)
    hist2,bins=np.histogram(algoStats['MLP'],bins=1000,range=(1,1000),density = True)
    hist2=np.cumsum(hist2)
    hist3,bins=np.histogram(algoStats['SVD'],bins=1000,range=(1,1000),density = True)
    hist3=np.cumsum(hist3)
    hist4,bins=np.histogram(algoStats['NeuMF'],bins=1000,range=(1,1000),density = True)
    hist4=np.cumsum(hist4)
    plt.figure(1)
    plt.plot(np.arange(0.1,2.1,0.1),hist4[0:20],'->k',label='NeuMF')
    plt.plot(np.arange(0.1,2.1,0.1),hist1[0:20],'-ob',label='GMF')
    plt.plot(np.arange(0.1,2.1,0.1),hist2[0:20],'-+r',label='MLP')
    plt.plot(np.arange(0.1,2.1,0.1),hist3[0:20],'-^m',label='SVD')
    plt.xticks(np.arange(0.1,2.1,0.1),rotation="vertical")
    plt.legend()
    plt.xlabel("Rank percentile of a top rated book among 1000 random books")
    plt.ylabel("Cumulative Distribution")
    plt.title("Comparison of NCF-Algorithms on Goodbooks-10k")
    plt.savefig("korenPlot1000.eps",format='eps')
    plt.figure(2)
    hist1,bins=np.histogram(algoStats_base['KNN'],bins=100,range=(1,100),density = True)
    hist1=np.cumsum(hist1)
    hist2,bins=np.histogram(algoStats_base['SVD'],bins=100,range=(1,100),density = True)
    hist2=np.cumsum(hist2)
    plt.plot(np.arange(1,21,1),hist1[0:20],'-ob',label='KNN')
    plt.plot(np.arange(1,21,1),hist2[0:20],'-+r',label='SVD')
    plt.legend()
    plt.xticks(range(1,21))
    plt.xlabel("Rank of a top rated book among 100 random books")
    plt.ylabel("Cumulative Distribution")
    plt.title("Comparison of classical Algorithms on Goodbooks-10k")
    plt.savefig("korenPlot100.eps",format="eps")

algoStats_base={}
with open ("NeuCFstats.json",'r') as inp:
    algoStats=json.load(inp)
with open ("algoStats_baseline.json",'r') as inp:
    st_base = json.load(inp)
algoStats['SVD']=st_base['svd']['top1000_rank']
algoStats_base['SVD']=st_base['svd']['top100_rank']
algoStats_base['KNN']=st_base['knn']['top100_rank']
koren_plot(algoStats,algoStats_base)
