import os
import numpy as np
import scipy.io as scio
from scipy import signal
import argparse

parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
parser.add_argument("-n", "--normal", default=False, action="store_true",help="normalization")
parser.add_argument("-e", "--eliminate_dc",default=False, action="store_true",help="eliminate dc")
parser.add_argument("-w", "--window_len", type=int, required=True,help="window length")
parser.add_argument("-o", "--overlap_len", type=int, required=True,help="overlap length")
args = parser.parse_args()


#sub_list = [1,2,3,4,5,6,7]
sub_list = [1]
SUB = len(sub_list)
GES = 4
TRI = 10

WINDOW_LEN = args.window_len
OVERLAP_LEN = args.overlap_len
TOTAL_LEN = 5000
err_mat = np.zeros([SUB,GES,TRI,TOTAL_LEN])
SLICE_NUM = (TOTAL_LEN - WINDOW_LEN) // OVERLAP_LEN + 1
Dataset = np.zeros([SUB,GES,TRI,SLICE_NUM,8,WINDOW_LEN])
valid_mat = np.zeros([SUB,GES,TRI])


def window_slice(data,window_len,overlap_len,sub,ges,tri):
    total_length = np.size(data,1)
    #segments = data[:,0:window_len]
    #segments = segments.reshape(1,8,window_len)
    segments = np.empty((0,8,window_len))

    time = 0 #incremental window
    while time + window_len <= total_length:
        segment_inc = data[:,time:time + window_len].reshape(1,8,window_len)
        mean = np.mean(segment_inc,2)
        #print("before",segment_inc)
        for i in range(8):
            segment_inc[0,i,:] = segment_inc[0,i,:] - mean[0,i]        
        #print("after",segment_inc)
        has_exception = False
        if np.sum(err_mat[sub,ges,tri,time:time+window_len]) != 0: #check whether has abrrupt change or not
            has_exception = True
        if has_exception == True:
            print('discard value in sub:', sub+1 , 'ges:',ges+1,'tri:',tri+1,'[',time,time+window_len,']') 
        else:
            segments = np.concatenate((segments,segment_inc),0)
        '''    
        max_value = np.max(segment_inc)
        #print(max_value)
        if max_value <= 3:
            segments = np.concatenate((segments,segment_inc),0)
        else:
            print('discard satured value in ', time , 'value:',max_value) #satured
        if time + window_len == total_length and np.min(segment_inc) < 0.04:
            print('discard small value in ', time , 'value:',np.min(segment_inc)) #satured
        '''
            

        time = time + overlap_len
    valid_mat[sub,ges,tri] = segments.shape[0]
    print(segments.shape)
    return segments

def exception_mark(data,sub,ges,tri):
    data_smooth = signal.savgol_filter(data,201,3)
    slope_mat = abs(data_smooth[:,0:TOTAL_LEN - 50] - data_smooth[:,50:TOTAL_LEN])
    for k in range(np.size(slope_mat,1)):
        if np.max(slope_mat[:,k]) > 0.12:
            err_mat[sub,ges,tri,k] = 1
    for k in range(50): #process last 50 slope
        if np.max(slope_mat[:,TOTAL_LEN - 50 - 50 + k]) > 0.12:
            err_mat[sub,ges,tri,TOTAL_LEN - 50 + k] = 1

def normalization(data):
    data_norm = np.zeros([8,TOTAL_LEN])
    for i in range(8):    
        max_s = np.max(data[i])
        min_s = np.min(data[i])
        data_norm[i] = (data[i] - min_s) / (max_s - min_s)
    return data_norm

for i in range(SUB):
    for j in range(GES):
        for k in range(TRI):
            mat_path = 'Dataset1/{:03d}_{:03d}_{:03d}.mat'.format(sub_list[i],j+1,k+1)
            print(i+1,j+1,k+1)
            if os.path.exists(mat_path):
                data = scio.loadmat(mat_path)['data'][:,0:TOTAL_LEN]
                if args.normal == True:
                    data = normalization(data)
                exception_mark(data,i,j,k)
                segments = window_slice(data,WINDOW_LEN,OVERLAP_LEN,i,j,k)
                valid_num = valid_mat[i,j,k]
                Dataset[i,j,k,0 : int(valid_num)] = segments
                #print(np.sum(err_mat[i,j,k,:9950]))
            else:
                pass
                

print(valid_mat)
np.save('valid_mat',valid_mat)
print(Dataset.shape)
np.save('Dataset',Dataset)

