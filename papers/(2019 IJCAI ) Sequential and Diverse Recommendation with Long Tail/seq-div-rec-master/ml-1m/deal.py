# Preprocess data
# make train, test, valid, dict_user, dict_item, clusterer, cid_items

import os
import numpy as np
import pandas as pd
import pickle
import collections
import random




cb_feat_size = 120
min_session_len = 2

####################################
#number of clusters, tail_cnt
####################################


tail_cnt = 300


save_path = '../temp/'
if not os.path.exists(save_path):
    os.mkdir(save_path)



###################################
#Make dictionary for item
###################################

class Dictionary(object):
    def __init__(self):
        self.item2idx = {}
        self.idx2item = []
    
    def add_item(self, item):
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item)-1
        return self.item2idx[item]


#Count support of each item


print('Make item dictionary')

dict_item = Dictionary()
dict_item_tail = Dictionary()

counter = collections.Counter()

#dictionary counter
print('Count the number of occurrence of each item..')

with open('../session.txt', 'r') as f:
    max_ses_len= 0
    line_len = 0
    for line in f:
        sess = line.split()
        sess = [int(item) for item in sess]
        if len(sess)>1: #ignore session with length 1
            max_ses_len = max(max_ses_len, len(sess))
            line_len += 1
            for item in sess:
                #dict_item.count(item)
                counter[item] += 1
              
                
#Top 10% of items are in head, the others are in tail
#tail_cnt = np.percentile(counter.values(), 90) # 22



for k, v in counter.items():
    if v > tail_cnt:
        dict_item.add_item(k)
    else:
        dict_item_tail.add_item(k)

        
print('num of head item:'+str(len(dict_item.idx2item))+ 'num of tail item: '+str(len(dict_item_tail.idx2item)))



###################################
#Make dictionary for user
###################################
print('Make user dictionary')
dict_user = Dictionary()

with open('../uid.txt', 'r') as f:
    for uid in f:
        uid = uid.replace('\n', '')
        dict_user.add_item(uid)

        
        
        
"""   
###################################
#Cluster tail items 
###################################

#Kmeans clustering
print 'Cluster tail items ..'
#clusterer = KMeans(n_clusters=n_cluster, random_state=0).fit(cb_feat_tail)
clusterer = MiniBatchKMeans(n_clusters=n_cluster, batch_size=n_cluster, init_size = 2*n_cluster).fit(cb_feat_tail)

#knn_graph = kneighbors_graph(cb_feat_tail, tail_cnt, include_self=True)
#clusterer = AgglomerativeClustering(linkage='ward', connectivity=knn_graph, n_clusters = n_cluster)
#clusterer.fit(cb_feat_tail)

#for clusterid in range(n_cluster):
#    rep_items = cb_items.most_similar(clusterer.cluster_centers_[clusterid], N=30)
#    rep_items = [i[0].replace('|', '_') for i in rep_items]
#    rep_id = [dict_item_tail.item2idx.get(i) for i in rep_items]
#    rep_id = [i for i in rep_id if i is not None]

#    print 'clusterid=', clusterid, [i.replace('_', '/') for i in rep_items if i in dict_item_tail.item2idx.keys()]


cid_items = {} # {cluster id, [item ids]}
for cid, item in zip(clusterer.labels_, dict_item_tail.idx2item):
    cid_items.setdefault(cid, []).append(item)

    
#add the clustered items into item dictionary
for clusterid in range(n_cluster):
    item = "cluster"+str(clusterid)
    dict_item.add_item(item)     
    
tail2cid={} # {item of tails, cid:idx}
for i, label in enumerate(clusterer.labels_):
    key = dict_item_tail.idx2item[i]
    item = "cluster"+str(label)
    idx = dict_item.item2idx[item]
    tail2cid[key] = idx
    
print 'num of items (head + clustered tail):', len(dict_item.idx2item)
    
#Save the clustering results
if not(os.path.exists(save_path+'clusterer.p')):
    pickle.dump(clusterer, open(save_path+'clusterer.p', 'wb'))

if not(os.path.exists(save_path+'cid_items.p')):
    pickle.dump(cid_items, open(save_path+'cid_items.p', 'wb'))
    
    
"""
###################################
#Make linked list of items for mini-batch
###################################   
class Item:
    def __init__(self,initdata, uid):
        self.data = initdata #item id
        self.next = None
        self.uid = uid #user id
        self.tail = None

def sort_freq(tail_list):
    counter = collections.Counter()
    for t in tail_list:
        counter[t] += 1
    
    #common_tail_list = counter.most_common()
    sorted_tail_list = sorted(counter, key=counter.get, reverse=True)
    
    return sorted_tail_list

#read data
print('Read data and convert it to list of linked lists for mini-batch')

isFirst=True
with open('../session.txt', 'r') as m, open('../uid.txt', 'r') as u:
    sessions = []
    for uid, line in zip(u, m):
        uid=uid.replace('\n','')
        uid_idx=dict_user.item2idx[uid]
        sess=line.split()
        sess = [int(item) for item in sess]
        if len(sess)>1:
            isFirst = True
            prev=None
            tail_list = []
            for item in sess:
                if item in dict_item.item2idx:
                    #head item
                    idx = dict_item.item2idx[item]
                    
                    cur = Item(idx, uid_idx)
                    cur.tail = sort_freq(tail_list)
                    tail_list = []
                    
                    
                    if isFirst:
                        sessions.append(cur)
                        isFirst = False
                    else:
                        prev.next =cur
                    prev = cur

                elif item in dict_item_tail.idx2item:
                    #tail item
                    idx = dict_item_tail.item2idx[item]
                    tail_list.append(idx)

print('num of sequences:'+str(len(sessions)))

#remove session with <2 items
def remove_short_session(source):
    short = []
    for i in range(len(source)):
        length = 0
        cur = source[i]
        while (cur is not None) and (length < min_session_len):
            length += 1
            cur = cur.next
        
        if length < min_session_len:
            short.append(i)
    
    for i in sorted(short, reverse = True):
        del source[i]
    #source = [s for i, s in enumerate(source) if i not in short]
    #return source
    
print('Remove sequence with <2 items')
remove_short_session(sessions)
print('num of sequences:'+str(len(sessions)))


# save preprocessed data
print('Save the preprocessed data: train.p, test.p, valid.p, dict_item.p, dict_user.p')
random.shuffle(sessions)
train_sz = int(0.7*len(sessions))
test_sz = int(0.2*len(sessions))
train = sessions[:train_sz]
test = sessions[train_sz: train_sz+test_sz]
valid= sessions[train_sz+test_sz: ]

import sys
sys.setrecursionlimit(10000)

if not(os.path.exists(save_path+'train.p')):
    pickle.dump(train, open(save_path+'train.p', 'wb'))

if not(os.path.exists(save_path+'test.p')):
    pickle.dump(test, open(save_path+'test.p', 'wb'))

if not(os.path.exists(save_path+'valid.p')):
    pickle.dump(valid, open(save_path+'valid.p', 'wb'))
    
if not(os.path.exists(save_path+'dict_item.p')):
    pickle.dump(dict_item, open(save_path+'dict_item.p', 'wb'))

if not(os.path.exists(save_path+'dict_user.p')):
    pickle.dump(dict_user, open(save_path+'dict_user.p', 'wb'))
    
    

    
    
    
###################################
#Preprocessing for evaluation: probability of being seen 
###################################  
print('Compute probability of items to be seen..')
uids_df = []
items_df = []

with open('../session.txt', 'r') as m, open('../uid.txt', 'r') as u:
    for uid, line in zip(u,m):
        uid = uid.replace('\n', '')
        uid_idx = dict_user.item2idx[uid]
        items = line.split()
        if len(items) > 1 :
            for item in items:
                uids_df.append(uid_idx)
                items_df.append(item)

df = pd.DataFrame(data={'uid':uids_df, 'item':items_df})
"""select item, count(distinct uid)
from df
group by item
"""
nuids = df.uid.nunique()
seen = df.groupby('item').uid.nunique()/nuids
#seen = - math.log(df.groupby('item').uid.nunique()/nuids,2)
seen = seen.to_dict() #{item_idx to p(seen)}


if not(os.path.exists(save_path+'seen.p')):
    pickle.dump(seen, open(save_path+'seen.p', 'wb'))

#seen is the frequency of item i being consumed by users

