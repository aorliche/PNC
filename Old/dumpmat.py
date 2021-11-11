
import scipy.io as sio
import pickle
#import sys

'''
mat = sio.loadmat('patient_list_imgset.mat')
print(mat['patient_list_imgset'][0])
'''

dir = '../../emoid_fmri_power264/timeseries/'

restMat = sio.loadmat('../../emoid_fmri_power264.mat')
#print(len(restMat['nback_fmri_power264'][0]['img_time_serie']))

#878 resting state
#910 nback
#680 emoid

'''
print(len(restMat['rest_fmri_power264'][0]['img_time_serie']))
print(type(restMat['rest_fmri_power264'][0]['img_time_serie'][0]))
quit()
'''

for subj in range(680):
    data = restMat['emoid_fmri_power264'][0]['img_time_serie'][subj]
    with open(dir + '/%d.bin' % subj, 'wb') as f:
        pickle.dump(data, f)
    print(subj, end=' ')
    
print('Complete')