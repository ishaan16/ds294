import json
import pandas as pd
import GMF
import MLP
import numpy as np
from surprise import *
from surprise.model_selection import train_test_split
import time

def getBooksNotRead(antiTrain,user_id):
    '''
    Returns a list of book_id's not rated by the user user_id
    '''
    bnr=set()
    for item in antiTrain:
        bnr.add(item[1])
    bnr = np.array(list(bnr))
    return bnr

def getTopKBooks(model,user,bnr,k=10):
    '''
    Returns top k predicted books for the user_id
    from among the books not in booksRated
    '''
    reco=np.zeros(k)
    l = len(bnr)
    users = np.full(l, user, dtype = 'int32')
    pred_ratings=model.predict([users,bnr],batch_size=100,verbose=1).flatten().tolist()
    pred_ratings = zip(bnr.tolist(),pred_ratings)
    reco = [item[0] for item in sorted(pred_ratings,key = lambda x:x[1],reverse=True)]
    return reco[:k+1]

def getTopKBooksBaselines(algo,user,bnr,k=10):
    reco=np.zeros(k)
    l = len(bnr)
    users = np.full(l, user, dtype = 'int32')
    pred_ratings=[]
    for item in bnr:
        pred_ratings.append((item,algo.predict(user,item).est))
    reco = [item[0] for item in sorted(pred_ratings,key = lambda x:x[1],reverse=True)]
    return reco[:k+1]
    
if __name__ == "__main__":
    r = pd.read_csv( 'ratings.csv' )
    num_books = r.book_id.max()
    num_users = r.user_id.max()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(r[['user_id', 'book_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2,random_state=8)
    anti_train = trainset.build_anti_testset() 
    topRecs = {}
    for user_id in [7799,45816]:
        bnr = getBooksNotRead(anti_train,user_id)
        print "Data extraction completed"
        topRecs[user_id]={}

        ########## SVD results ########
        algo2=SVD(n_factors=100,random_state=3)
        algo2.fit(trainset)
        topRecs[user_id]['SVD']= getTopKBooksBaselines(algo2,user_id,bnr,k=30)
        sim1 = {'name': 'pearson_baseline',
                'user_based': False,  # compute  similarities between items
                'shrinkage':0}
        algo1 = KNNBaseline(k=40,min_k=1,sim_options=sim1)
        algo1.fit(trainset)
        topRecs[user_id]["KNN"]=getTopKBooksBaselines(algo1,user_id,bnr,k=30)
        
        ######## GMF results ##########
        model = GMF.get_model(num_users,num_books,8)
        modelFile = "Pretrain/gb-10k_GMF_8_trainset.h5"
        model.load_weights(modelFile)
        topRecs[user_id]['GMF']=getTopKBooks(model,user_id,bnr,30)

        ######### MLP results ##############
        model2 = MLP.get_model(num_users,num_books,layers=[64,32,16,8],reg_layers=[0,0,0,0])
        modelFile = "Pretrain/gb-10k_MLP_[64,32,16,8]_trainset.h5"
        model2.load_weights(modelFile)
        topRecs[user_id]['GMF']=getTopKBooks(model2,user_id,bnr,30)

    with open ('recommend.json','w') as inp:
        json.dump(topRecs,inp)
    
