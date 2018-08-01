# You need to write down your own code here
# Task: Given any head entity name (e.g. Q30) and relation name (e.g. P36), you need to output the top 10 closest tail entity names.
# File entity2vec.vec and relation2vec.vec are 50-dimensional entity and relation embeddings.
# If you use the embeddings learned from Problem 1, you will get extra credits.
import pickle as pkl
import tensorflow as tf
import numpy as np
import pandas as pd

z = []
_x = open('entity2id.txt', 'r').readlines()[1:]
for x in _x:
    z.append(x.split())
e2is = pd.DataFrame(z)
e2is.columns = ['name', 'code']
e2is.iloc[:,1] = e2is.iloc[:,1].astype('int64')


z = []
_x = open('relation2id.txt', 'r').readlines()[1:]
for x in _x:
    z.append(x.split())
r2is = pd.DataFrame(z)
r2is.columns = ['name', 'code']
r2is.iloc[:,1] = r2is.iloc[:,1].astype('int64')

r2v = np.loadtxt('relation2vec.txt')
e2v = np.loadtxt('entity2vec.txt')

## Question 1

q1_h = 'Q30'
q1_r = 'P36'

q1_h_num = e2is[e2is['name'] == q1_h].code.values[0]
q1_r_num = r2is[r2is['name'] == q1_r].code.values[0]

q1_h_v = e2v[q1_h_num]
q1_r_v = r2v[q1_r_num]


q1_t_norm = np.linalg.norm(q1_h_v + q1_r_v - e2v, ord=1, axis=1)
q1_top_10 = pd.DataFrame(q1_t_norm).reset_index().sort_values(0)[:10]
print("\n\n\nQuestion1")
print("Top 10 smallest energy function score:")
for i, code in enumerate(q1_top_10['index'].values):
    print i+1, e2is[e2is.code==code].name.values
    
print("The correct answer for capital of US - \"Washington, D.C.\" (Q61) is in the 7th place.")


## Question 2

q2_h = 'Q30'
q2_t = 'Q49'

q2_h_num = e2is[e2is['name'] == q2_h].code.values[0]
q2_t_num = e2is[e2is['name'] == q2_t].code.values[0]

q2_h_v = e2v[q2_h_num]
q2_t_v = e2v[q2_t_num]


q2_r_norm = np.linalg.norm(q2_h_v - q2_t_v - r2v, ord=1, axis=1)
q2_top_10 = pd.DataFrame(q2_r_norm).reset_index().sort_values(0)[:10]

print("\n\n\nQuestion2")
print("Top 10 smallest energy function score:")
for i, code in enumerate(q2_top_10['index'].values):
    print i+1, r2is[r2is.code==code].name.values
    
print("The top-1 relation is P1336 - \"territory claimed by\"")