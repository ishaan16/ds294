import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
import numpy as np

r = pd.read_csv( 'ratings.csv' )
num_books = r.book_id.max()
num_users = r.user_id.max()
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(r[['user_id', 'book_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2,random_state=8)
i=1
with open ("Data/gb-10k.train.rating",'w') as inp:
    for item in trainset.all_ratings():
        us = trainset.to_raw_uid(item[0])
        bk = trainset.to_raw_iid(item[1])
        inp.write("%d\t%d\t%1.1f\n"%(us,bk,item[2]))


