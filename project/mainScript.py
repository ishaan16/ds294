import os

#Split the ratings into test and training sets
os.execfile('python test_train.py')

# Run Baseline codes
os.execfile('python baselines.py')

#Run NeuCF code
os.execfile('python GMF.py --dataset gb-10k --epochs 15 --batch_size 256 --num_factors 8 --regs [0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1')

os.execfile('python MLP.py --dataset gb-10k --epochs 15 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1')

os.execfile('python NeuMF.py --dataset gb-10k --epochs 15 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1 --mf_pretrain Pretrain/gb-10k_GMF_8_trainset.h5 --mlp_pretrain Pretrain/gb-10k_MLP_[64,32,16,8]_trainset.h5')

#Evaluate the results
os.execfile('python koren_eval.py')

#Plot the results
os.execfile('python plotResults.py')

#Check the top recommendations
os.execfile('python topRecommend.py')
