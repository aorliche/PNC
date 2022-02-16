import mat73
import pickle
import numpy as np
from nilearn.connectome import ConnectivityMeasure

matName = 'emoid_fmri_power264'
metaName = 'MegaMetaEmoid2.pkl'
para = 'emoid'
doWrite = True

validEthnicities = ['CAUCASIAN/WHITE','ASIAN','AFRICAN','OTHER/MIXED','HAWAIIAN/PACIFIC','AMERICAN']

meta = dict()
mat = mat73.loadmat(f'{matName}.mat')
mm = mat[matName]

cm = ConnectivityMeasure(kind='correlation')

for i in range(len(mm['meta'])):
    met = mm['meta'][i]
    key = int(met['id'])
    age = int(met['age_in_month'])
    gender = met['gender']
    ethnicity = met['ethnicity']
    if age < 8*12 or age > 23*12:
        print(f'Alert age {key}')
    if gender != 'M' and gender != 'F':
        print(f'Alert gender {key}')
    if ethnicity not in validEthnicities:
        print(f'Alert ethnicity {ethnicity} {key}')
    ts = mm['img_time_serie'][i].astype('float32')
    fc = cm.fit_transform([ts.T])
    if np.sum(np.isnan(fc)) > 0:
        print(f'Alert fc {key}')
    if i % 50 == 0:
        print(f'Done {i}')
    meta[key] = {
        'AgeInMonths': age,
        'Gender': gender,
        'Ethnicity': ethnicity,
        para: ts
    }
    
if len(meta.keys()) != len(mm['meta']):
    print(f'Alert, key length {len(meta.keys())} != {len(mm["meta"])}')

if doWrite:
    print(f'Writing pickle to {metaName}')
    with open(metaName, 'wb') as f:
        pickle.dump(meta, f)

print('Complete')