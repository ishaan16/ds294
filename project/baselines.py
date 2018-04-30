import pandas as pd
import numpy as np
from surprise import *
import time
from surprise.model_selection import *
import matplotlib.pyplot as plt

def rankOf(key,arr):
    val = arr[key]
    arrSort = sorted(arr)
    v = len(arr)
    for i in xrange(v-1,-1,-1):
        if arrSort[i]<=val:
            return v-i
    return 0

def rankInTopK(algo,k=100):
    #Requires global variables trainset, testset, num_books
    i=0
    top_rated_rank=[]
    t=time.time()
    for entry in testset:
        if entry[2]==5.0: 
            book = entry[1]
            user = entry[0]
            i+=1
            negs=[book]
            while book in negs:
                negs=np.random.choice(np.arange(1,num_books+1),k-1)
            negs=np.append(negs,book)
            pred_ratings=[]
            for item in negs:
                pred_ratings.append(algo.predict(user,item).est)
            top_rated_rank.append(rankOf(k-1,pred_ratings))
    print ("Time taken in hours=%1.3f"%((time.time()-t)/3600))
    return top_rated_rank

def rmseMeasure(algo):
    #Requires global variables testset
    predictions = algo.test(testset)
    return(accuracy.rmse(predictions))

if __name__=="__main__":
    algoStats={}
    r = pd.read_csv( 'ratings.csv' )
    num_users = r.user_id.max()
    num_books = r.book_id.max()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(r[['user_id', 'book_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2,random_state=8)

    algoStats['svd']={}
    algo2=SVD(n_factors=100,random_state=3)
    algo2.fit(trainset)
    algoStats["svd"]={'rmse':rmseMeasure(algo1),
                      'top1000_rank':rankInTopK(algo2,k=1000),
                      'top100_rank':rankInTopK(algo2,k=100)}
    
    algoStats["knn"]={}
    sim1 = {'name': 'pearson_baseline',
            'user_based': False,  # compute  similarities between items
            'shrinkage':0}
    algo1 = KNNBaseline(k=40,min_k=1,sim_options=sim1)
    algo1.fit(trainset)
    algoStats["knn"]={'rmse':rmseMeasure(algo1),
                      'top100_rank':rankInTopK(algo1,k=100)}
    with open("algoStats_baseline.json") as inp:
        json.dump(algoStats,inp)
