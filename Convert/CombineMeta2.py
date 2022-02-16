import csv
import sys
import pickle

sys.path.append('..')

from LoadData import loadMeta

pncDir = '../../PNC_Good'
outputFile = f'{pncDir}/MegaMeta2.pkl'

# Read wrat

wratFile = f'{pncDir}/wrat2.csv'
wratRaw = dict()
wratStd = dict()

with open(wratFile, newline='') as f:
	reader = csv.reader(f, delimiter=',')
	first = True
	for row in reader:
		if first:
			first = False
			continue
		try:
			key = int(row[0])
			wratRaw[key] = int(row[2])
			wratStd[key] = int(row[3])
		except Exception:
			print(f'Alert {row}')

print(len(wratRaw.keys()))
print(len(wratStd.keys()))

# Read meta files

restMeta = loadMeta(f'{pncDir}/MegaMetaRest2.pkl')
nbackMeta = loadMeta(f'{pncDir}/MegaMetaNback2.pkl')
emoidMeta = loadMeta(f'{pncDir}/MegaMetaEmoid2.pkl')

# Combine meta files

meta = dict()

def mergeInto(big, small):
	for key in small:
		if key in big:
			for subkey in small[key]:
				if subkey in big[key]:
					if big[key][subkey] != small[key][subkey]:
						print(f'Alert {key} {subkey}')
				else:
					big[key][subkey] = small[key][subkey]
		else:
			big[key] = small[key]

mergeInto(meta, restMeta)
mergeInto(meta, nbackMeta)
mergeInto(meta, emoidMeta)

print('After merge')

print(len(meta.keys()))
print(len(restMeta.keys()))
print(len(nbackMeta.keys()))
print(len(emoidMeta.keys()))

'''
testKey = list(meta.keys())[0]
print(meta[testKey])
print(restMeta[testKey])
print(nbackMeta[testKey])
print(emoidMeta[testKey])
'''

# Add in wrat

count = 0
for key in meta.keys():
	if key in wratStd:
		count += 1
		meta[key]['wratRaw'] = wratRaw[key]
		meta[key]['wratStd'] = wratStd[key]

print(f'{count} subjects have WRAT')

# Write to file

with open(outputFile, 'wb') as f:
	pickle.dump(meta, f)

print('Complete')
