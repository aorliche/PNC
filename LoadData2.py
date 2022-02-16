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
	if scanType in entry:
		return True

def allScansIncFn(entry, scanTypes):
	return functools.reduce(lambda yes, scanType: yes and scanIncFn(entry, scanType), 
		scanTypes, True)

def scanLoadFn(entry, scanType):
	return entry[scanType]

def genderLoadFn(entry, labels):
	return labels[entry['Gender']]

def ageLoadFn(entry):
	return entry['AgeInMonths']/12

def loadNbackEmoidIdsScansAndGenders(meta):
	incFn = lambda key, entry: allScansIncFn(entry, ['nback','emoid'])
	kLoadFn = lambda key, entry: key
	nLoadFn = lambda key, entry: scanLoadFn(entry, 'nback')
	eLoadFn = lambda key, entry: scanLoadFn(entry, 'emoid')
	gLoadFn = lambda key, entry: genderLoadFn(entry, {'M':0, 'F':1})
	keys = loadFromMeta(meta, incFn, kLoadFn)
	nback = loadFromMeta(meta, incFn, nLoadFn)
	emoid = loadFromMeta(meta, incFn, eLoadFn)
	genders = loadFromMeta(meta, incFn, gLoadFn)
	return keys, np.stack(nback), np.stack(emoid), np.array(genders, dtype='long')

def loadNbackEmoidScansAndAges(meta):
	incFn = lambda key, entry: allScansIncFn(entry, ['nback','emoid'])
	kLoadFn = lambda key, entry: key
	nLoadFn = lambda key, entry: scanLoadFn(entry, 'nback')
	eLoadFn = lambda key, entry: scanLoadFn(entry, 'emoid')
	aLoadFn = lambda key, entry: ageLoadFn(entry)
	keys = loadFromMeta(meta, incFn, kLoadFn)
	nback = loadFromMeta(meta, incFn, nLoadFn)
	emoid = loadFromMeta(meta, incFn, eLoadFn)
	ages = loadFromMeta(meta, incFn, aLoadFn)
	return keys, np.stack(nback), np.stack(emoid), np.array(ages)

def getFC(timeSeries, kind='correlation', transpose=True):
	connMeasure = ConnectivityMeasure(kind=kind)
	if transpose:
		timeSeries = np.transpose(timeSeries, axes=(0,2,1))
	return connMeasure.fit_transform(timeSeries)
