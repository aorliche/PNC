import itertools
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.dense import DenseGCNConv

import numpy as np

def normalize(A):
    if A.shape[0] != A.shape[1]:
        raise Exception("Bad A shape")
    d = torch.sum(A,dim=1)**0.5
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

def trainGCN(gcn, trainA, trainFeat, trainLabels, nEpochs=2000, pPeriod=200, lr=1e-5, wd=0.2, verbose=True):
	optim = torch.optim.Adam(gcn.parameters(), lr=lr, weight_decay=wd)

	for epoch in range(nEpochs):
		optim.zero_grad()
		pred = gcn([trainA,trainFeat]).flatten()
		loss = gcn.loss(pred, trainLabels)
		loss.backward()
		optim.step()
		if (epoch % pPeriod == 0 or epoch == nEpochs-1) and verbose:
			print(f'epoch {epoch} loss={loss}')

	print(f'Complete GCN') if verbose else None

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

def trainCG(cg, trainFeat, trainLabels, nEpochs=30, bSize=1000, pPeriod=2000, lr=2e-5, wd=0, verbose=True):
	optim = torch.optim.Adam(cg.parameters(), lr=lr, weight_decay=wd)
	pairs = list(itertools.combinations_with_replacement(np.arange(trainFeat.shape[0]),2))

	print(f'Training for {nEpochs} epochs') if verbose else None
	for epoch in range(nEpochs):
		randPairs = copy.copy(pairs)
		random.shuffle(randPairs)
		nComplete = 0

		print(f'epoch {epoch}') if verbose else None
		while nComplete < len(pairs):
			todo = len(pairs)-nComplete
			if todo > bSize:
				todo = bSize
			batchPairs = randPairs[nComplete:nComplete+todo]
			Ai, Bi = zip(*batchPairs)
			A = trainFeat[Ai,:]
			B = trainFeat[Bi,:]
			a = trainLabels[list(Ai)]
			b = trainLabels[list(Bi)]
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
			if nComplete % pPeriod == 0 and verbose:
				print(f'\tposLoss={pLoss} negLoss={nLoss}')
			nComplete += todo

	print(f'Completed {nEpochs*len(pairs)} comparisons') if verbose else None

def evalCG(cg, trainFeat, trainLabels, testFeat, testLabels=None, pPeriod=20, verbose=True):
	N = testFeat.shape[0]
	Ntrain = trainFeat.shape[0]

	if testLabels:
		assert N == testLabels.shape[0], "testFeat and testLabels have different lengths"
		g = torch.zeros(trainFeat.shape[1])
	else:
		wp = np.zeros(N)
		wn = np.zeros(N)

	print(f'Evaluating {N} samples') if verbose else None
	for i in range(N):
		if i % pPeriod == 0 and verbose:
			print(f'done {i}')
			
		A = testFeat[i].expand(Ntrain,-1)
		B = trainFeat
		b = trainLabels

		if testLabels:
			cg.zero_grad()

		pos = torch.cat([A,B],dim=1)
		neg = torch.cat([B,A],dim=1)
		pdelta = cg(pos).flatten()
		ndelta = cg(neg).flatten()
		pres = torch.mean(pdelta + b)
		nres = torch.mean(b - ndelta)

		if testLabels:
			loss = cg.loss((pres+nres)/2, testLabels[i])
			loss.backwards()
			g += cg.layers[1].weight.grad@cg.layers[0].weight
		else:
			wp[i] = pres.detach().cpu().numpy()
			wn[i] = nres.detach().cpu().numpy()
		
	print('Complete') if verbose else None
	return g if testLabels else wp,wn

if __name__ == '__main__':
	cg = createCG([100,1], torch.nn.MSELoss())

	from LoadData2 import loadNbackEmoidAgesScansAndGenders, loadMeta

	pncDir = '../PNC_Good'

	keys, nbackTs, emoidTs, ages = loadNbackEmoidAgesScansAndGenders(loadMeta(f'{pncDir}/MegaMeta3.pkl'))

	print(nbackTs.shape)
	print(emoidTs.shape)
	print(ages.shape)
	print(ages[0:10])

	# Get FC and convert to torch

	from LoadData2 import getFC
	import torch

	nbackP = getFC(nbackTs)
	emoidP = getFC(emoidTs)

	nbackP_t = torch.from_numpy(nbackP).reshape(650,264*264).float().cuda()
	emoidP_t = torch.from_numpy(emoidP).reshape(650,264*264).float().cuda()
	feat_t = torch.cat([nbackP_t, emoidP_t], dim=1)
	ages_t = torch.from_numpy(ages).float().cuda()

	print(nbackP_t.shape)
	print(emoidP_t.shape)
	print(ages_t.shape)

	import numpy as np

	for i in range(1):
		si = i*100
		ti = (i+1)*100
		ei = (i+2)*100
		trainFeat = feat_t[si:ti]
		testFeat = feat_t[ti:ei]
		feat = feat_t[si:ei]
		trainA = cosineSimilarityAdjacency(trainFeat, trainFeat)
		A = cosineSimilarityAdjacency(feat, feat)
		trainLabels = ages_t[si:ti]
		testLabels = ages_t[ti:ei]
		testLabels_np = testLabels.detach().cpu().numpy()
		w, _, _, _ = torch.linalg.lstsq(trainFeat, trainLabels)
		pred = testFeat@w
		pred = pred.detach().cpu().numpy()
		#gcn = createGCN([2*264*264,100,1], torch.nn.MSELoss())
		#trainGCN(gcn, trainA, trainFeat, trainLabels, nEpochs=2000, verbose=False)
		#pred = gcn([A, feat]).flatten()[(ti-si):].detach().cpu().numpy()
		print(np.mean((pred-testLabels_np)**2)**0.5)
		#cg = createCG([4*264*264,100,1], torch.nn.MSELoss())
		#trainCG(cg, trainFeat, trainLabels, verbose=False)
		#wp,wn = evalCG(cg, trainFeat, trainLabels, testFeat, verbose=False)
		#print(np.mean((wp-testLabels_np)**2)**0.5)
		#print(np.mean((wn-testLabels_np)**2)**0.5)
