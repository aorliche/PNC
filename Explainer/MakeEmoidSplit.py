import pickle
import numpy as np
import math

with open('../../PNC/AllSubjectsMeta.bin', 'rb') as f:
	meta = pickle.load(f)

print(len(list(meta.keys())))

para = 'emoid'
blacklist = [180]

idsEmoid = np.array([key for key in meta if para in meta[key] and int(meta[key][para]) not in blacklist])

agesEmoid = []

for key in meta.keys():
	if para in meta[key] and int(meta[key][para]) not in blacklist:
		agesEmoid.append(meta[key]['meta']['AgeInMonths']/12)

agesEmoid = np.array(agesEmoid)
nTrain = math.floor(0.9*len(idsEmoid))

perm = np.random.permutation(len(idsEmoid))
trainIdx = idsEmoid[perm[:nTrain]]
testIdx = idsEmoid[perm[nTrain:]]
trainAges = np.array(agesEmoid[perm[:nTrain]])
testAges = np.array(agesEmoid[perm[nTrain:]])

print(len(trainIdx))
print(len(testIdx))

with open('../../Work/Explainer/EmoidSplit1.bin', 'wb') as f:
	pickle.dump({'trainIds': trainIdx, 'testIds': testIdx}, f)

with open('../../Work/Explainer/EmoidSplit1.bin', 'rb') as f:
	print(pickle.load(f))

with open('../../Work/Explainer/EmoidSplit1Ages.bin', 'wb') as f:
	pickle.dump({'trainAges': trainAges, 'testAges': testAges}, f)

with open('../../Work/Explainer/EmoidSplit1Ages.bin', 'rb') as f:
	print(pickle.load(f))
