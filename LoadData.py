from nilearn.connectome import ConnectivityMeasure
import pickle
import functools
import numpy as np

def loadMeta(metaPath):
	with open(metaPath, 'rb') as f:
		return pickle.load(f)

def loadFromMeta(meta, incFn, loadFn, intersect=True):
	data = []
	for key in meta:
		if incFn(key, meta[key]):
			data.append(loadFn(key, meta[key]))
	return data

def scanIncFn(entry, scanType):
	badKey = 'bad'+scanType.capitalize()
	if scanType in entry and badKey not in entry:
		return True

def allScansIncFn(entry, scanTypes):
	return functools.reduce(lambda yes, scanType: yes and scanIncFn(entry, scanType), 
		scanTypes, True)

def scanLoadFn(entry, scanType):
	return entry[scanType+'Data']

def genderLoadFn(entry, labels):
	return labels[entry['meta']['Gender']]

def loadEmoidIdsScansAndGenders(meta):
	incFn = lambda key, entry: scanIncFn(entry, 'emoid')
	kLoadFn = lambda key, entry: key
	sLoadFn = lambda key, entry: scanLoadFn(entry, 'emoid')
	gLoadFn = lambda key, entry: genderLoadFn(entry, {'M':0, 'F':1})
	keys = loadFromMeta(meta, incFn, kLoadFn)
	scans = loadFromMeta(meta, incFn, sLoadFn)
	genders = loadFromMeta(meta, incFn, gLoadFn)
	return keys, np.stack(scans), np.array(genders, dtype='long')

def getFC(timeSeries, kind='correlation'):
	connMeasure = connectome.ConnectivityMeasure(kind=kind)
	return connMeasure.fit_transform(timeSeries)

if __name__ == '__main__':
	keys, scans, genders = loadEmoidIdsScansAndGenders(loadMeta('../../PNC/MegaMeta.bin'))
	scans = np.transpose(scans, axes=(0,2,1))
	print(scans.shape)
	print(genders.shape)
	print(genders[0:10])
	pearsonFC = getFC(scans)
	print(pearsonFC.shape)
	partialFC = getFC(scans, kind='partial correlation')
	print(partialFC.shape)
