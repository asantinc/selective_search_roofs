import os
import shutil
import glob
import cv2
import pdb
import numpy as np

import sklearn.utils
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

import utils 


class RoofLoader(object):
    def load_images(self, labels_tuples, max_roofs):
        '''
        Load the images into a numpy array X.
        '''
        X = np.empty((max_roofs,3,utils.PATCH_W, utils.PATCH_H))
        file_names = list()
        img_to_remove = list()
        labels = list()
        failures = 0

        index = 0
        for i, (f_name, roof_type) in enumerate(labels_tuples):
            if i%1000 == 0:
                print 'Loading image {0}'.format(i)
            f_number = int(f_name)
            f_path = utils.FTRAIN+str(f_number)+'.jpg'                
            x = cv2.imread(f_path)
            x = np.asarray(x, dtype='float32')/255
            try:
                x = x.transpose(2,0,1)
                x.shape = (1,x.shape[0], x.shape[1], x.shape[2])
                #X = x if X==None else np.concatenate((X, x), axis=0)
                X[index, :, :, :] = x
            except ValueError, e:
                print e
                failures += 1
                print 'fail:'+ str(failures)
            else:
                index += 1
                file_names.append(f_number)
                labels.append(roof_type)
        X = X.astype(np.float32)
        return X[:index, :,:,:], labels



###################################################
## Load image patches for neural network.
# 1. Loads positives and negatives (non_roofs*number of positive)
# 2. The scaling has already been done in get_data
# 3. Here we do the division over 255 though
###################################################
#    def neural_load_detections_from_viola(self, in_path=utils.TRAINING_NEURAL_PATH):
        #Need to load
        # the false positives and true positives for thatch AND metal
#        raise ValueError() 


    def neural_load_training(self, train_path=None, neg_path=None, max_roofs=None, non_roofs=None):
        '''
        Load all the positive training examples for the neural network and labels to a numpy array
        '''
        raise ValueError("TODO: need to fix these paths")
        train_path = utils.TRAINING_NEURAL_POS if train_path is None else train_path
        neg_path = utils.TRAINING_NEURAL_NEG if neg_path is None else neg_path
        train_labels =  np.loadtxt(open(train_path+'labels.csv',"rb"),delimiter=",")
        X_train, y_train = self.get_neural_data(train_labels, train_path)
        
        neg_labels = np.loadtxt(open(neg_path+'labels.csv', 'rb'),delimiter=',')
        pdb.set_trace()
        X_neg_train, y_neg_train = self.get_neural_data(neg_labels[:y_train.shape[0]*non_roofs],neg_path)
       
        print X_train.shape, y_train.shape
        print X_neg_train.shape, y_neg_train.shape
        pdb.set_trace()
        X = np.concatenate((X_train, X_neg_train), axis=0)
        labels = np.concatenate((y_train, y_neg_train), axis=0)

        #shuffle data
        X, labels = sklearn.utils.shuffle(X, labels, random_state=42)  # shuffle train data    
        return X, labels


    def get_neural_data(self, label_tuples, path):
        X = np.empty((len(label_tuples), 3, utils.PATCH_W, utils.PATCH_H))
        labels = list()
        failures = 0
        index = 0

        for i, (f_name, roof_type) in enumerate(label_tuples):
            if i%1000 == 0:
                print 'Loading image {0}'.format(i)
            f_number = int(f_name)
            f_path = path+str(f_number)+'.jpg'                
            
            x = cv2.imread(f_path)
            x = np.asarray(x, dtype='float32')/255
            try:
                x = x.transpose(2,0,1)
                x.shape = (1,x.shape[0], x.shape[1], x.shape[2])
                X[index, :, :, :] = x
            except ValueError, e:
                print e
                failures += 1
                print 'fail:'+ str(failures)
            else:
                index += 1
                labels.append(roof_type)
  
        #remove any failed images
        if failures > 0:
            X = X[:-failures, :,:,:] 

        #return the right type
        X = X.astype(np.float32)
        labels = np.array(labels).astype(np.int32)
        return X, labels


if __name__ == "__main__":
    loader = RoofLoader()
    loader.load(max_roofs=5000)

