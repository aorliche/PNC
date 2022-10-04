# Using newly preprocessed subjects

import pickle

import numpy as np

from scipy import signal

import math
import threading
import itertools

metadictname = 'PNC_agesexwrat.pkl'
alltsname = 'PNC_PowerTS_float2.pkl'

with open(metadictname, 'rb') as f:
    metadict = pickle.load(f)

with open(alltsname, 'rb') as f:
    allts = pickle.load(f)
    
print(list(metadict.keys()))
print(list(allts.keys()))
print('Loading Complete')

'''
Get subjects that have all tasks and paras specified
Functions for creating independent and response variables
'''

def get_subs(allts, metadict, tasks, paras):
    # Get subs for all paras
    for i,para in enumerate(paras):
        tmpset = set([int(sub[4:]) for sub in allts[para].keys()])
        if i == 0:
            paraset = tmpset
        else:
            paraset = paraset.intersection(tmpset)
    # Get subs for all tasks
    for i,task in enumerate(tasks):
        tmpset = set([sub for sub in metadict[task].keys()])
        if i == 0:
            taskset = tmpset
        else:
            taskset = paraset.intersection(tmpset)
    # Remove QC failures
    allsubs = taskset.intersection(paraset)
    for badsub in metadict['failedqc']:
        try:
            allsubs.remove(int(badsub[4:]))
        except:
            pass
    return allsubs

def get_X(allts, paras, subs):
    X = []
    for para in paras:
        pX = [allts[para][f'sub-{sub}'] for sub in subs]
        pX = np.stack(pX)
        X.append(pX)
    return X

def get_y(metadict, tasks, subs):
    y = []
    for task in tasks:
        if task == 'age' or task == 'wrat':
            var = [metadict[task][sub] for sub in subs]
            var = np.array(var)
            y.append(var)
        if task == 'sex':
            maleness = [metadict[task][sub] == 'M' for sub in subs]
            maleness = np.array(maleness)
            sex = np.stack([maleness, 1-maleness], axis=1)
            y.append(sex)
    return y

subs = get_subs(allts, metadict, ['age'], ['rest', 'nback', 'emoid'])
print(len(subs))

X = get_X(allts, ['rest'], subs)
print(X[0].shape)
print('Got timeseries')

# TS to condensed FC

def butter_bandpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = [cutoff[0] / nyq, cutoff[1] / nyq]
    b, a = signal.butter(order, normal_cutoff, btype='band', analog=False)
    return b, a

def butter_bandpass_filter(data, cutoff, fs, order=5):
    b, a = butter_bandpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

tr = 1.83
N = X[0].shape[0]

def filter_design_ts(X):
    Xs = []
    for i in range(X.shape[0]):
        nX = butter_bandpass_filter(X[i], [tr/20*N, 0.8*N], 2*N)
        Xs.append(nX)
    return np.stack(Xs)

def ts_to_flat_fc(X):
    p = np.corrcoef(X)
    a,b = np.triu_indices(p[0].shape[0], 1)
    p = p[a,b]
    return p

ts = [ts for ts in filter_design_ts(X[0])]
ts = np.stack(ts)
print(ts.shape)
print('Filtered timeseries')

def makepoly(x, order):
    N = x.shape[-1]
    idcs = itertools.combinations(range(N),order)
    idcs = [np.array(comb) for comb in idcs]
    for power in range(1,order+1):
        idcs += [np.array(power*[idx]) for idx in range(N)]
    xp = [np.prod(x[:,idx], axis=1) for idx in idcs]
    xp = np.array(xp).T
    return xp

# print(makepoly(np.expand_dims(np.arange(4), 0),2))
# raise 'bad'

def tfun(x, gc, sub, nt, nw):
    i,j = np.meshgrid(np.arange(264),np.arange(264))
    i = i.reshape(-1)
    j = j.reshape(-1)
    # Granger input
    A = [makepoly(np.concatenate([x[sub,i,a:a+nw],x[sub,j,a:a+nw]], axis=1),2) for a in range(0,nt-nw)]
    A = np.stack(A, axis=1)
    # Only previous nw values
    B = [makepoly(x[sub,i,a:a+nw],2) for a in range(0,nt-nw)]
    B = np.stack(B, axis=1)
    # Target value
    C = x[sub,i,nw:nt]
    Aplus = np.linalg.pinv(A)
    Bplus = np.linalg.pinv(B)
    wij = np.einsum('nat,nt->na',Aplus,C)
    wi = np.einsum('nat,nt->na',Bplus,C)
    Cij = np.einsum('nta,na->nt',A,wij)
    Ci = np.einsum('nta,na->nt',B,wi)
    eij = C-Cij
    ei = C-Ci
    sigmaij = np.var(eij, axis=1)
    sigmai = np.var(ei, axis=1)
    gcsingle = np.log(sigmai/sigmaij)
    gc[sub] = gcsingle
    if sub % 10 == 0:
        print(f'Finished {sub}')

nt = ts.shape[-1]
nw = 1
x = ts
gc = ts.shape[0]*[None]
n_thread = 20

print('Beginning granger processing')

for batch in range(math.floor(x.shape[0]/n_thread)+1):
    nleft = x.shape[0] - batch*n_thread
    todo = n_thread if nleft > n_thread else nleft
    threads = []
    for batchsub in range(todo):
        sub = batch*n_thread+batchsub
        t = threading.Thread(target=tfun, args=(x, gc, sub, nt, nw))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

gc = np.stack(gc)
gc = gc.reshape(gc.shape[0],264,264)
print('Granger Complete')

with open(f'gc{nw}byorder2rest_ageall3.pkl', 'wb') as f:
    pickle.dump(gc, f)

print('Write Granger Complete')
