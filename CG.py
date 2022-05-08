import itertools
import random
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.dense import DenseGCNConv

import numpy as np
import scipy.stats as stats

def normalize(A):
	if A.shape[0] != A.shape[1]:
		raise Exception("Bad A shape")
	d = F.relu(torch.sum(A,dim=1))**0.5
	d[d == 0] = 1
	return ((A/d).T/d).T

def cosineSimilarity(a, b):
    e = torch.einsum('ai,bi->ab',a,b)
    aa = torch.einsum('ai,ai->a',a,a)**0.5
    bb = torch.einsum('bi,bi->b',b,b)**0.5
    e /= aa.unsqueeze(1)
    e /= bb.unsqueeze(1).T
    return e

def cosineSimilarityAdjacency(a, b):
	return normalize(cosineSimilarity(a,b)-torch.eye(a.shape[0]).float().cuda())

def createGCN(layerSizes, lossFn):
	class GCN(nn.Module):
		def __init__(self):
			super(GCN, self).__init__()
			self.layers = []
			for i in range(len(layerSizes)-1):
				self.layers.append(DenseGCNConv(layerSizes[i], layerSizes[i+1]).float().cuda())
			self.layers = nn.Sequential(*self.layers)
			self.loss = lossFn

		def forward(self, x):
			A = x[0]
			x = x[1]
			for layer in self.layers:
				x = F.relu(layer(x,A)) if layer != self.layers[-1] else layer(x,A)
			return x
	return GCN()

def trainGCN(gcn, trainA, trainFeat, trainLabels, nEpochs=2000, pPeriod=200, lr=2e-5, wd=0.2, verbose=True,
		regNotClass=True):
	optim = torch.optim.Adam(gcn.parameters(), lr=lr, weight_decay=wd)

	for epoch in range(nEpochs):
		optim.zero_grad()
		if regNotClass:
			pred = gcn([trainA,trainFeat]).flatten()
		else:
			pred = gcn([trainA,trainFeat]).squeeze()
		loss = gcn.loss(pred, trainLabels)
		loss.backward()
		optim.step()
		if (epoch % pPeriod == 0 or epoch == nEpochs-1) and verbose:
			print(f'epoch {epoch} loss={loss}')

	print(f'Complete GCN') if verbose else None

def createMLP(layerSizes, lossFn):
	class MLP(nn.Module):
		def __init__(self):
			super(MLP, self).__init__()
			self.layers = []
			for i in range(len(layerSizes)-1):
				self.layers.append(nn.Linear(layerSizes[i], layerSizes[i+1]).float().cuda())
			self.layers = nn.Sequential(*self.layers)
			self.loss = lossFn

		def forward(self, x):
			for layer in self.layers:
				x = F.relu(layer(x)) if layer != self.layers[-1] else layer(x)
			return x
	return MLP()

def trainMLP(mlp, trainFeat, trainLabels, nEpochs=2000, pPeriod=200, lr=2e-5, wd=0.2, verbose=True,
		regNotClass=True):
	optim = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)

	for epoch in range(nEpochs):
		optim.zero_grad()
		if regNotClass:
			pred = mlp(trainFeat).flatten()
		else:
			pred = mlp(trainFeat).squeeze()
		loss = mlp.loss(pred, trainLabels)
		loss.backward()
		optim.step()
		if (epoch % pPeriod == 0 or epoch == nEpochs-1) and verbose:
			print(f'epoch {epoch} loss={loss}')

	print(f'Complete MLP') if verbose else None

def createCG(layerSizes, lossFn):
	class CG(nn.Module):
		def __init__(self):
			super(CG, self).__init__()
			self.layers = []
			for i in range(len(layerSizes)-1):
				self.layers.append(nn.Linear(layerSizes[i], layerSizes[i+1]).float().cuda())
			self.layers = nn.Sequential(*self.layers)
			self.loss = lossFn

		def forward(self, x):
			for layer in self.layers:
				x = F.relu(layer(x)) if layer != self.layers[-1] else layer(x)
			return x
	return CG()

def getLabels(trainLabels, i, j, N):
	t = trainLabels[:N].detach().float()
	m = t == j
	t[m] = 1
	t[torch.logical_not(m)] = 0
	t[m] *= N/torch.sum(t)
	return t, m

def trainCG2(cg, trainFeat, trainLabels, nEpochs=50, bSize=1000, lr=2e-5, wd=0, verbose=True, 
		alpha=1, beta=1, gamma=1, chgMult=1):
	N = trainLabels.shape[0]
	nper = math.floor(bSize/N)

	assert nper > 0
	
	numClasses = torch.max(trainLabels)+1
	optim = torch.optim.Adam(cg.parameters(), lr=lr, weight_decay=wd)
	Nchg = chgMult*int(bSize**0.5)
	Nstop = N if N < Nchg else Nchg
	
	if verbose:
		print(f'Training for {nEpochs} epochs')

	for epoch in range(nEpochs):
		epLoss = 0
		epLoss2 = 0
		for n in range(0,Nstop,nper):
			optim.zero_grad()
			sz = min(nper,N-n)
			A = trainFeat[n:(n+nper)]
			B = trainFeat[:Nstop]
			res = torch.zeros(sz,numClasses).float().cuda()
			sav = torch.zeros(sz,Nstop).float().cuda()
			tgt = torch.zeros(sz,Nstop).float().cuda()
			for i in range(A.shape[0]):
				inpP = torch.cat([A[i].expand(Nstop,-1),B],dim=1)
				inpN = torch.cat([B,A[i].expand(Nstop,-1)],dim=1)
				pos = cg(inpP).flatten()
				neg = cg(inpN).flatten()
				r = (pos+neg)/2
				sav[i] = r
				for j in range(numClasses):
					t, m = getLabels(trainLabels, n+i, j, Nstop)
					res[i,j] = r@t
					if trainLabels[n+i] == j:
						tgt[i] = m
			loss = cg.loss(res, trainLabels[n:(n+nper)])
			loss2 = alpha*torch.mean((sav-tgt)**2)
			(loss+loss2).backward()
			optim.step()
			epLoss += loss.detach()/Nstop*sz
			epLoss2 += loss2.detach()/Nstop*sz
		optim.zero_grad()
		inp = torch.cat([B,B],dim=1)
		r = cg(inp).flatten()
		cMean = torch.zeros(numClasses).float().cuda()
		for i in range(numClasses):
			m = trainLabels[:Nstop] == i
			cMean[i] = torch.mean(r[m])
		loss3 = beta*torch.std(cMean)
		loss4 = gamma*torch.mean((r-1)**2)
		(loss3+loss4).backward()
		optim.step()
		if verbose:
			print(f'epoch={epoch} loss={epLoss} loss2={epLoss2/alpha} loss3={loss3/beta} loss4={loss4/gamma}')
		if Nstop == N and epLoss < 0.03 and epLoss2/alpha < 0.03 and loss3/beta < 0.03:
			if verbose:
				print('Early stopping')
			break
		if Nstop < N and epLoss < 0.1 and epLoss2/alpha < 0.1 and loss3/beta < 0.1:
			Nstop *= 2
			if Nstop > N:
				Nstop = N
			if verbose:
				print(f'Changing Nstop {Nstop}')

	if verbose:
		print(f'Completed {epoch+1} {N*N} comparisons')
	
	#print(cg(torch.cat([trainFeat, trainFeat], dim=1)).flatten())

def evalCG2(cg, trainFeat, trainLabels, testFeat, testLabels=None):
	N = trainLabels.shape[0]
	numClasses = torch.max(trainLabels)+1
	if testLabels is not None:
		assert N == testLabels.shape[0], "testFeat and testLabels have different lengths"
		g = torch.zeros(trainFeat.shape[1]).float().cuda()
	else:
		res = torch.zeros(testFeat.shape[0], numClasses).float().cuda()
	for i in range(testFeat.shape[0]):
		B = nn.Parameter(testFeat[i], requires_grad=True)
		Be = B.expand(N,-1)
		inpP = torch.cat([Be,trainFeat],dim=1)
		inpN = torch.cat([trainFeat,Be],dim=1)
		pos = cg(inpP).flatten()
		neg = cg(inpN).flatten()
		r = (pos+neg)/2
		rr = torch.zeros(numClasses).float().cuda()
		for j in range(numClasses):
			t, m = getLabels(trainLabels, i, j, N)
			rr[j] = r@t
		if testLabels is not None:
			loss = cg.loss(rr, testLabels[i])
			loss.backward()
			g += F.relu(B.grad*B).detach()
		else:
			res[i] = rr.detach()
	if testLabels is not None:
		return g
	else:
		unc = 1/(0.1+torch.std(res, axis=1))
		unc = unc.detach().cpu().numpy()
		res = torch.argmax(res, dim=1)
		res = res.detach().cpu().numpy()
		return res, unc

def trainCG(cg, trainFeat, trainLabels, nEpochs=50, bSize=1000, pPeriod=1000, lr=2e-5, wd=0, verbose=True):
	N = trainLabels.shape[0]
	allIdcs = torch.arange(N).long().cuda()
	pairs = list(itertools.combinations_with_replacement(np.arange(N),2))
	optim = torch.optim.Adam(cg.parameters(), lr=lr, weight_decay=wd)
	
	if verbose:	
		print(f'Training for {nEpochs} epochs')

	for epoch in range(nEpochs):
		randPairs = copy.copy(pairs)
		random.shuffle(randPairs)
		nComplete = 0

		if verbose:
			print(f'epoch {epoch}')

		for n in range(0,N,bSize):
			batchPairs = randPairs[n:n+bSize]
			Ai, Bi = zip(*batchPairs)
			Ai = torch.tensor(list(Ai)).long().cuda()
			Bi = torch.tensor(list(Bi)).long().cuda()
			A = trainFeat[Ai,:]
			B = trainFeat[Bi,:]
			a = trainLabels[Ai]
			b = trainLabels[Bi]
			optim.zero_grad()
			pos = torch.cat([A,B],dim=1)
			neg = torch.cat([B,A],dim=1)
			pres = cg(pos).flatten()
			nres = cg(neg).flatten()
			pp = pres-(a-b)
			nn = nres-(b-a)
			pLoss = cg.loss(pp, torch.zeros(todo).float().cuda())
			nLoss = cg.loss(nn, torch.zeros(todo).float().cuda())
			(pLoss+nLoss).backward()
			optim.step()
			if n % pPeriod == 0 and verbose:
				print(f'\tpLoss={pLoss} nLoss={nLoss}')

	if verbose:
		print(f'Completed {nEpochs*len(pairs)} comparisons') if verbose else None

def evalCG(cg, trainFeat, trainLabels, testFeat, testLabels=None):
	N = testFeat.shape[0]
	Ntrain = trainFeat.shape[0]

	if testLabels is not None:
		assert N == testLabels.shape[0], "testFeat and testLabels have different lengths"
		g = torch.zeros(trainFeat.shape[1]).float().cuda()
	else:
		wp = np.zeros(N)
		wn = np.zeros(N)
		up = np.zeros(N)
		un = np.zeros(N)

	for i in range(N):
		cg.zero_grad()
		part = nn.Parameter(testFeat[i], requires_grad=True)		
		A = part.expand(Ntrain,-1)
		B = trainFeat
		b = trainLabels
		pos = torch.cat([A,B],dim=1)
		neg = torch.cat([B,A],dim=1)
		pdelta = cg(pos).flatten()
		ndelta = cg(neg).flatten()
		pres = torch.mean(pdelta + b)
		nres = torch.mean(b - ndelta)

		if testLabels is not None:
			loss = cg.loss((pres+nres)/2, testLabels[i])
			loss.backward()
			g += F.relu(part.grad*part).detach()
		else:
			wp[i] = pres.detach().cpu().numpy()
			wn[i] = nres.detach().cpu().numpy()
			up[i] = torch.std(pdelta).detach().cpu().numpy()
			un[i] = torch.std(ndelta).detach().cpu().numpy()

	if testLabels is not None:
		return g
	else:
		return wp,wn,up,un

if __name__ == '__main__':
	cg = createCG([100,1], torch.nn.MSELoss())

	from LoadData2 import loadNbackEmoidScansAgesAndGenders, loadMeta

	pncDir = '../PNC_Good'

	keys, nbackTs, emoidTs, genders, ages = loadNbackEmoidScansAgesAndGenders(
		loadMeta(f'{pncDir}/MegaMeta3.pkl'))

	print(nbackTs.shape)
	print(emoidTs.shape)
	#print(ages.shape)
	#print(ages[0:10])

	# Get FC and convert to torch

	from LoadData2 import getFC
	import torch

	nbackP = getFC(nbackTs)
	emoidP = getFC(emoidTs)

	ages_t = torch.from_numpy(ages).unsqueeze(1).expand(650,1000).float().cuda()
	nbackP_t = torch.from_numpy(nbackP).reshape(650,264*264).float().cuda()
	emoidP_t = torch.from_numpy(emoidP).reshape(650,264*264).float().cuda()
	feat_t = torch.cat([nbackP_t, emoidP_t], dim=1)
	genders_t = torch.from_numpy(genders).cuda()

	print(nbackP_t.shape)
	print(emoidP_t.shape)
	print(genders_t.shape)

	import numpy as np
	from sklearn.linear_model import LogisticRegression

	setSize = 30
	for i in range(0,1):
		si = i*setSize
		ti = (i+1)*setSize
		ei = (i+2)*setSize
		'''
		idcs = np.arange(si,ti)
		midcs = np.intersect1d(np.where(genders == 0)[0], idcs)
		fidcs = np.intersect1d(np.where(genders == 1)[0], idcs)
		m = min(midcs.shape[0], fidcs.shape[0])
		idcs = torch.from_numpy(np.concatenate([midcs,fidcs])).cuda()
		'''
		trainFeat = feat_t[si:ti]
		testFeat = feat_t[ti:ei]
		feat = torch.cat([trainFeat, testFeat])
		trainA = cosineSimilarityAdjacency(trainFeat, trainFeat)
		A = cosineSimilarityAdjacency(feat, feat)
		trainLabels = genders_t[si:ti]
		trainLabels_np = trainLabels.detach().cpu().numpy()
		testLabels = genders_t[ti:ei]
		testLabels_np = testLabels.detach().cpu().numpy()
		'''
		#w, _, _, _ = torch.linalg.lstsq(trainFeat, trainLabels)
		#pred = testFeat@w
		#pred = pred.detach().cpu().numpy()
		print(f'Start {i}')
		clf = LogisticRegression(max_iter=1000).fit(trainFeat.detach().cpu().numpy(), trainLabels_np)
		pred = clf.predict(testFeat.detach().cpu().numpy())
		print(np.sum(pred == testLabels_np))
		print(pred.astype(int))
		mlp = createMLP([trainFeat.shape[1],100,2], torch.nn.CrossEntropyLoss())
		trainMLP(mlp, trainFeat, trainLabels, nEpochs=1000, verbose=False, regNotClass=False, wd=0)
		pred = torch.argmax(mlp(testFeat).squeeze(), dim=1).detach().cpu().numpy()
		print(np.sum(pred == testLabels_np))
		print(pred.astype(int))
		gcn = createGCN([trainFeat.shape[1],100,2], torch.nn.CrossEntropyLoss())
		trainGCN(gcn, trainA, trainFeat, trainLabels, nEpochs=1000, verbose=False, regNotClass=False)
		pred = torch.argmax(gcn([A, feat]).squeeze(), dim=1)[trainFeat.shape[0]:].detach().cpu().numpy()
		print(np.sum(pred == testLabels_np))
		print(pred.astype(int))
		#print(np.mean((pred-testLabels_np)**2)**0.5)
		continue
		'''
		def getMat(cg, train, test):
			both = torch.cat([train, test], dim=0)
			N = both.shape[0]
			mat = np.zeros([N,N])
			for i in range(N):
				A = both[i].expand(N, -1)
				pos = torch.cat([A, both], dim=1)
				neg = torch.cat([both, A], dim=1)
				pr = cg(pos).flatten()
				nr = cg(neg).flatten()
				mat[i,:] = ((pr+nr)/2).detach().cpu().numpy()
				mat[:,i] = mat[i,:]
			return mat

		cg = createCG([2*trainFeat.shape[1],100,1], torch.nn.CrossEntropyLoss())
		trainCG2(cg, trainFeat, trainLabels, lr=5e-6, wd=0, nEpochs=1000, alpha=500, beta=50, gamma=10, chgMult=2)
		res, unc = evalCG2(cg, trainFeat, trainLabels, trainFeat)
		print(np.sum(res == trainLabels_np))
		mat = getMat(cg, trainFeat, testFeat)
		print(mat[0,:])
		print(mat[1,:])
		res = evalCG2(cg, trainFeat, trainLabels, testFeat)
		print(np.sum(res == testLabels_np))
		print(res)
		print(testLabels_np)
