import json
import NeuMF
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import MLP
import GMF
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
import time

def rankOf(key,arr):
    val = arr[key]
    arrSort = sorted(arr)
    v = len(arr)
    for i in xrange(v-1,-1,-1):
        if arrSort[i]<=val:
            return v-i
    return 0
def rankInTopK(model,testset,num_books,k=100):
    #Requires global variables testset, num_books
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
            users = np.full(k, user, dtype = 'int32')
            pred_ratings=model.predict([users,negs],batch_size=k,verbose=0)
            top_rated_rank.append(rankOf(k-1,pred_ratings))
	    if i%10000 == 0:
		print("%d iterations done"%i) 
    print ("Time taken in hours=%1.3f"%((time.time()-t)/3600))
    return top_rated_rank

if __name__ == "__main__":
    r = pd.read_csv( 'ratings.csv' )
    num_books = r.book_id.max()
    num_users = r.user_id.max()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(r[['user_id', 'book_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2,random_state=8)
    algoStats={}
    ######## MLP results ##########
    model2 = MLP.get_model(num_users,num_books,layers=[64,32,16,8],reg_layers=[0,0,0,0])
    modelFile = "Pretrain/gb-10k_MLP_[64,32,16,8]_trainset.h5"
    model2.load_weights(modelFile)
    algoStats["MLP"] = rankInTopK(model2,testset,num_books,k=1000)
    ######## GMF results ##########
    model = GMF.get_model(num_users,num_books,8)
    modelFile = "Pretrain/gb-10k_GMF_8_trainset.h5"
    model.load_weights(modelFile)
    algoStats["GMF"] = rankInTopK(model,testset,num_books,k=1000)
    ######## NeuMF results ##########
    model3 = NeuMF.get_model(num_users, num_books, mf_dim=8, layers=[64,32,16,8], reg_layers=[0,0,0,0], reg_mf=0)
    modelFile = "Pretrain/gb-10k_NeuMF_8_[64,32,16,8]_trainset.h5"
    model3.load_weights(modelFile)
    algoStats["NeuMF"] = rankInTopK(model3,testset,num_books,k=1000)

    with open ("NeuCFstats.json",'w') as inp:
        json.dump(algoStats,inp)
