# -*- coding: utf-8 -*-
import models
import utils
import collections
import random
import pandas as pd
import argparse
import os
import torch
import numpy as np
import time
import pickle
import random

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import itertools

# from aurochs.buffalo import feature
cb_feat_size = 120
min_session_len = 2

####################################
# number of clusters, tail_cnt
####################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tail_cnt = 0

save_path = 'temp/'
if not os.path.exists(save_path):
    os.mkdir(save_path)


###################################
# Make dictionary for item
###################################

class Dictionary(object):
    def __init__(self):
        self.item2idx = {}
        self.idx2item = []

    def add_item(self, item):
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item) - 1
        return self.item2idx[item]


# Count support of each item


print('Make item dictionary')

dict_item = Dictionary()
dict_item_tail = Dictionary()

counter = collections.Counter()

# dictionary counter
print('Count the number of occurrence of each item..')

with open('session.txt', 'r') as f:
    max_ses_len = 0
    line_len = 0
    for line in f:
        sess = line.split()
        sess = [int(item) for item in sess]
        if len(sess) > 1:  # ignore session with length 1
            max_ses_len = max(max_ses_len, len(sess))
            line_len += 1
            for item in sess:
                # dict_item.count(item)
                counter[item] += 1

# Top 10% of items are in head, the others are in tail
# tail_cnt = np.percentile(counter.values(), 90) # 22


for k, v in counter.items():
    if v > tail_cnt:
        dict_item.add_item(k)
    else:
        dict_item_tail.add_item(k)

print('num of head item:' + str(len(dict_item.idx2item)) + 'num of tail item: ' + str(len(dict_item_tail.idx2item)))

###################################
# Make dictionary for user
###################################
print('Make user dictionary')
dict_user = Dictionary()

with open('uid.txt', 'r') as f:
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
# Make linked list of items for mini-batch
###################################
class Item:
    def __init__(self, initdata, uid):
        self.data = initdata  # item id
        self.next = None
        self.uid = uid  # user id
        self.tail = None


def sort_freq(tail_list):
    counter = collections.Counter()
    for t in tail_list:
        counter[t] += 1

    # common_tail_list = counter.most_common()
    sorted_tail_list = sorted(counter, key=counter.get, reverse=True)

    return sorted_tail_list


# read data
print('Read data and convert it to list of linked lists for mini-batch')

isFirst = True
with open('session.txt', 'r') as m, open('uid.txt', 'r') as u:
    sessions = []
    for uid, line in zip(u, m):
        uid = uid.replace('\n', '')
        uid_idx = dict_user.item2idx[uid]
        sess = line.split()
        sess = [int(item) for item in sess]
        if len(sess) > 1:
            isFirst = True
            prev = None
            tail_list = []
            for item in sess:
                if item in dict_item.item2idx:
                    # head item
                    idx = dict_item.item2idx[item]

                    cur = Item(idx, uid_idx)
                    cur.tail = sort_freq(tail_list)
                    tail_list = []

                    if isFirst:
                        sessions.append(cur)
                        isFirst = False
                    else:
                        prev.next = cur
                    prev = cur

                elif item in dict_item_tail.idx2item:
                    # tail item
                    idx = dict_item_tail.item2idx[item]
                    tail_list.append(idx)

print('num of sequences:' + str(len(sessions)))


# remove session with <2 items
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

    for i in sorted(short, reverse=True):
        del source[i]
    # source = [s for i, s in enumerate(source) if i not in short]
    # return source


print('Remove sequence with <2 items')
remove_short_session(sessions)
print('num of sequences:' + str(len(sessions)))

# save preprocessed data
print('Save the preprocessed data: train.p, test.p, valid.p, dict_item.p, dict_user.p')
random.shuffle(sessions)
train_sz = int(0.7 * len(sessions))
test_sz = int(0.2 * len(sessions))
train = sessions[:train_sz]
test = sessions[train_sz: train_sz + test_sz]
valid = sessions[train_sz + test_sz:]
print('Compute probability of items to be seen..')
uids_df = []
items_df = []

with open('session.txt', 'r') as m, open('uid.txt', 'r') as u:
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















print ('Ready to train...')
bsz = 1024 # batch size
eval_bsz = 1024
cuda = True # Enable GPU or not


ninp=500 # size of embedding
nhid=100 # size of hidden layers
nlayers=1 # number of GRU
log_interval = 1000
epochs = 1

cb_feat_size = 120
topk=20

#k=6 # num. of clusters
#tail_cnt = np.percentile(dict_item.cnt.values(), 90) # 22



######################################################
#Which model to train?
######################################################

GRU4REC = 0
GRU4REC_RERANKING = 1
GRU4REC_CB = 2
GRU4DIV = 3
GRU4DIV_CB = 4

# parser = argparse.ArgumentParser(description = 'Select 0:gru4rec, 1:gru4rec+reranking, 2:gru4recCB, 3:gru4div, 4:gru4divCB')
# parser.add_argument('model', metavar = 'model', type=int,default=3)

# args = parser.parse_args()
# which_model = args.model
which_model = 3

######################################################
#Load data
######################################################

#Each preprocess data is type of a list of items. Item is linked to next item.
class Dictionary(object):
    def __init__(self):
        self.item2idx = {}
        self.idx2item = []

    def add_item(self, item):
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item)-1
        return self.item2idx[item]

class Item:
    def __init__(self,initdata, uid):
        self.data = initdata #item id
        self.next = None
        self.uid = uid #user id
        self.tail = None

#preprocessed data

nitem = len(dict_item.idx2item)




#
# #######################################################
# #Contents vector
# ######################################################
# ROOT = 'data/2017092711-cb-text/'
# cb_items = feature.load(os.path.join(ROOT, 'main'))

# cb_feat = []
# for item in dict_item.idx2item:
#     item = item.replace("_", "|")
#
#     if 'cluster' in item:
#         cid = int(item[7:])
#         feat = clusterer.cluster_centers_[cid]
#     else:
#         feat = cb_items.feats[cb_items.idmap[item]]
#
#     cb_feat.append(feat)
#
# cb_feat = np.array(cb_feat)


#######################################################
#Build model
######################################################
name = 'model_'
if (which_model == GRU4REC):
    model = models.GRU4rec(nitem, ninp, nhid)
    criterion = utils.ListMLE_loss()
    name +='GRU4rec'

elif (which_model == GRU4REC_RERANKING):
    name += 'GRU4rec_reranking'

elif (which_model == GRU4REC_CB):
    model = models.GRU4recCB(nitem, ninp, nhid, cb_feat_size)
    criterion = utils.ListMLE_loss()
    name += 'GRU4rec_cb'

elif (which_model == GRU4DIV):
    print ('Preparing model...')
    model = models.GRU4rec(nitem, ninp, nhid)
    criterion = utils.ListMLE_loss_tail(cuda)
    print ('model preparation finished...')
    print (model)
    for para in model.parameters():
        print (para.shape)
    name += 'GRU4div'

elif (which_model == GRU4DIV_CB):
    model = models.GRU4recCB(nitem, ninp, nhid, cb_feat_size)
    criterion = utils.ListMLE_loss_tail(cuda)
    name +='GRU4div_cb'
else:
    print ('Error!')

print ('start model deployment...')
if cuda:
    model = model.to(device)
print ('model deployed on GPU successfully')
torch.autograd.set_detect_anomaly(True)
#######################################################
#Train
######################################################
def training():
    model.train() # turn on training mode which enables dropout
    # iterate over batch
    total_ranking_loss = 0

    start_time = time.time()
    #Initial hidden state is input for our model
    hidden = model.init_hidden(bsz)

    batch = utils.get_batch(train, bsz, cuda)
    data = []
    iteration = 0

    optimizer = optim.Adagrad(model.parameters())
    optimizer.zero_grad()

    while True:
        data, target, tails, init= next(batch, None) #gives None if the generator expires instead of StopIteration error

        if data is None:
            break

        # Starting each  batch, we detach the hidden state from how it was previous produced.
        # If we didnt' the model would try backpropagating all the way to start of the dataset


        print('data shape:{}'.format(data.shape))
        print('target shape:{}'.format(target.shape))
        #reset hidden state for indepedent new session
        for i in init:
            hidden.data[0][i,:].zero_()



        output, hidden = model(data,hidden)

        loss, temp = criterion(output.view(-1, nitem), target, tails)

        loss = torch.mean(loss)
        print('Loss calculation finished...')
        loss.backward()
        print('Gradient calculation finished...')



        #'clip_grad_norm' helps prevent the exploding gradient problem in RNN
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)

        #SGD
        #for p in model.parameters():
        #    p.data.add_(-lr, p.grad.data)

        #Adagrad
        optimizer.step()
        optimizer.zero_grad()

        total_ranking_loss += loss.data

        iteration += 1

        if iteration % log_interval == 0 and iteration > 0:
            cur_ranking_loss = total_ranking_loss[0] / log_interval
            elapsed = time.time()-start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss_ranking {:8.5f} '.format(epoch, iteration, len(train)//bsz,  elapsed * 1000/log_interval, cur_ranking_loss))

            total_ranking_loss = 0
            start_time=time.time()


# def trainingCB(fixed_params):
#     model.train() # turn on training mode which enables dropout
#     # iterate over batch
#     total_ranking_loss = 0
#
#     start_time = time.time()
#     #Initial hidden state is input for our model
#     hidden = model.init_hidden(bsz)
#     hidden_cb = model.init_hidden(bsz)
#
#     batch = utils.get_batch(train, bsz, cuda)
#     data = []
#     iteration = 0
#
#     for param in model.parameters():
#         param.requires_grad = True
#
#     optimizer = optim.Adagrad(model.parameters())
#
#     i=0
#     for param in model.parameters():
#         if i in fixed_params:
#             param.requires_grad = False
#         else:
#             param.requires_grad = True
#         i+=1
#
#
#     model.gru.flatten_parameters()
#     model.gru_cb.flatten_parameters()
#
#     while True:
#         data, target, tails, init= next(batch, None) #gives None if the generator expires instead of StopIteration error
#
#         if data is None:
#             break
#
#         # Starting each  batch, we detach the hidden state from how it was previous produced.
#         # If we didnt' the model would try backpropagating all the way to start of the dataset
#         hidden = utils.repackage_hidden(hidden)
#         hidden_cb = utils.repackage_hidden(hidden_cb)
#         doc_emb = utils.get_doc_emb(data, cb_feat, cb_feat_size, cuda)                      #输入数据与处理函数中描述不同
#
#         #reset hidden state for indepedent new session
#         for i in init:
#             hidden.data[0][i,:].zero_()
#             hidden_cb.data[0][i,:].zero_()
#
#         optimizer.zero_grad()
#
#         output, hidden, hidden_cb = model(data, doc_emb, hidden, hidden_cb)                 #data is a list of item_id tensors
#
#         loss, temp = criterion(output.view(-1, nitem), target, tails)
#
#         loss = torch.mean(loss)
#
#         loss.backward()
#
#
#         #'clip_grad_norm' helps prevent the exploding gradient problem in RNN
#         torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
#
#         #SGD
#         #for p in model.parameters():
#         #    p.data.add_(-lr, p.grad.data)
#
#         #Adagrad
#         optimizer.step()
#
#         total_ranking_loss += loss.data
#
#         iteration += 1
#
#         if iteration % log_interval == 0 and iteration > 0:
#             cur_ranking_loss = total_ranking_loss[0] / log_interval
#             elapsed = time.time()-start_time
#             print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss_ranking {:8.5f} '.format(epoch, iteration, len(train)//bsz,  elapsed * 1000/log_interval, cur_ranking_loss))
#
#             total_ranking_loss = 0
#             start_time=time.time()
#


#######################################################
# Evaluate
#######################################################
def evaluate(source):
    model.eval()

    hidden = model.init_hidden(eval_bsz)
    data = []
    batch = utils.get_batch(source, eval_bsz, cuda)
    iteration = 0
    total_ranking_loss =0

    #accuracy (head)
    total_acc_h = np.zeros(4) #mrr, map, ndcg, precision

    #accuracy (head + tail)
    total_acc_ht = np.zeros(4)

    #novelty
    total_p_unseen = 0

    #aggregate diversity
    total_rec_set = set()

    #individual diversity
    total_distance = 0
    model.gru.flatten_parameters()

    while True:
        data, target, tails, init = next(batch, None)
        if data is None:
            break

        iteration +=1

        #reset hidden state for indepedent new session
        for i in init:
            hidden.data[0][i,:].zero_()

        data = Variable(data.data, volatile=True)
        output, hidden = model(data,hidden)

        #Loss
        loss, temp = criterion(output.view(-1, nitem), target, tails)
        loss = torch.mean(loss)

        total_ranking_loss +=loss.data

        #topk
        _, indices = torch.topk(output.view(-1, nitem).data, topk)
        idx = indices.cpu().numpy()
        target = target.data.cpu().numpy()

        #Accuracy with only head
        acc_h = utils.get_accuracy(idx, target, topk)
        total_acc_h += acc_h
        #Accuracy with head and tail
        acc_ht = utils.get_accuracy_tail(idx, target, tails, topk)
        total_acc_ht += acc_ht


        #Long-tail measures
        # p_unseen, rec_set, distance = utils.get_long_tail_measures(idx, dict_item, eval_bsz, cb_items, cid_items, seen, topk )
        # total_p_unseen += p_unseen
        # total_rec_set = total_rec_set.union(rec_set)
        # total_distance += distance


    return total_ranking_loss[0]/iteration, total_acc_h/iteration, total_acc_ht/iteration, total_p_unseen/iteration#, total_rec_set, total_distance/iteration




# def evaluateCB(source):
#     model.eval()
#
#     hidden = model.init_hidden(eval_bsz)
#     hidden_cb = model.init_hidden(eval_bsz)
#     data = []
#     batch = utils.get_batch(source, eval_bsz, cuda)
#     iteration = 0
#     total_ranking_loss =0
#
#     #accuracy (head)
#     total_acc_h = np.zeros(4) #mrr, map, ndcg, precision
#
#     #accuracy (head + tail)
#     total_acc_ht = np.zeros(4)
#
#     #novelty
#     total_p_unseen = 0
#     #aggregate diversity
#     total_rec_set = set()
#     #individual diversity
#     total_distance = 0
#     model.gru.flatten_parameters()
#     model.gru_cb.flatten_parameters()
#
#
#     while True:
#         data, target, tails, init = next(batch, None)
#         if data is None:
#             break
#
#         iteration +=1
#
#         #reset hidden state for indepedent new session
#         for i in init:
#             hidden.data[0][i,:].zero_()
#             hidden_cb.data[0][i,:].zero_()
#
#         doc_emb = utils.get_doc_emb(data, cb_feat, cb_feat_size, cuda)
#         output, hidden, hidden_cb = model(data, doc_emb, hidden, hidden_cb)
#
#         #Loss
#         loss, temp = criterion(output.view(-1, nitem), target, tails)
#         loss = torch.mean(loss)
#
#         total_ranking_loss +=loss.data
#
#
#         #topk
#         _, indices = torch.topk(output.view(-1, nitem).data, topk)
#         idx = indices.cpu().numpy()
#         target = target.data.cpu().numpy()
#
#         #Accuracy
#         #mrr, mAP, ndcg, precision = get_accuracy(output, target)
#         acc_h = utils.get_accuracy(idx, target, topk)
#         total_acc_h += acc_h
#         #Accuracy with tail
#         acc_ht = utils.get_accuracy_tail(idx, target, tails, topk)
#         total_acc_ht += acc_ht
#
#
#         #Long-tail measures
#         p_unseen, rec_set, distance = utils.get_long_tail_measures(idx, dict_item, eval_bsz, cb_items, cid_items, seen, topk )
#         total_p_unseen += p_unseen
#         total_rec_set = total_rec_set.union(rec_set)
#         total_distance += distance
#
#     return total_ranking_loss[0]/iteration, total_acc_h/iteration, total_acc_ht/iteration, total_p_unseen/iteration, total_rec_set, total_distance/iteration
#



########################################################
#Is there any good way of automaticallyd detecting the structure and fixing parameters?
#num_param= 0
#for param in model.parameters():
#    print(num_param, param.size())
#    num_param += 1
subnet_param = [5,6,7,8] #those parameter tensor will be fixed during optimization
subnet_param_cb = [0,1,2,3,4]


#######################################################
#Run train
######################################################
best_val_loss =None
#logs = []

#Loop over epochs
#At any point you can hit Ctrl + C to break out of training early
print('Starting...')
try:
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()

        # if (which_model == GRU4REC_CB) or (which_model == GRU4DIV_CB):
        #     #alternating optimization
        #
        #     # trainingCB(subnet_param)
        #     # trainingCB(subnet_param_cb)
        #     # val_ranking_loss, acc_h, acc_ht, unseen, rec_set, distance= evaluateCB(valid)
        #
        # else:
        print ('epoch ' + str(epoch) + ' start training')
        training()
        print ('epoch ' + str(epoch) + ' start evaluating')
        val_ranking_loss, acc_h, acc_ht, unseen= evaluate(valid)



        print('-'*89)
        print('| end of epoch {:3d} | time: {:8.5f}s | valid ranking loss {:8.5f} '.format(epoch, (time.time() - epoch_start_time), val_ranking_loss))
        print (acc_h, acc_ht, unseen)#, len(rec_set), distance #acc = [mrr, map, ndcg, precision]
        print('-'*89)

        #logs.append([val_ranking_loss, acc_h, acc_ht, unseen, len(rec_set), distance])

        #save the model if the validation loss is the best we've seen so far
        #if not best_val_loss or val_ranking_loss < best_val_loss:
        with open(name+'.p', 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_ranking_loss
except KeyboardInterrupt:
    print('-'*89)
    print('Exiting from training early')



######################################################
#Load the best saved model
######################################################
with open(name+'.p', 'rb') as f:
    model = torch.load(f)



#####################################################
#Run on test data
######################################################

#test1 = test[:len(test)/2]
#test2 = test[len(test)/2:]


# if (which_model == GRU4REC_CB) or (which_model == GRU4DIV_CB):
#     # test_ranking_loss, acc_h, acc_ht, unseen, rec_set, distance= evaluateCB(test)
#
# else:
    test_ranking_loss, acc_h, acc_ht, unseen, rec_set, distance= evaluate(test)

print('='*89)
print('| end of epoch {:3d} | time: {:8.5f}s | valid ranking loss {:8.5f} '.format(epoch, (time.time() - epoch_start_time), test_ranking_loss))
print ( acc_ht, unseen, len(rec_set), distance) #acc = [mrr, map, ndcg, precision]
print('='*89)
