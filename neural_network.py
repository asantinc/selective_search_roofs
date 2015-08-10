import os
import sys
sys.setrecursionlimit(10000) #so you can pickle large nets

import subprocess
import getopt
import numpy as np
import sys
import pdb
import csv
from operator import itemgetter
import cv2
import cPickle as pickle

import lasagne
import theano
sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')
from nolearn.lasagne.handlers import PrintLog, PrintLayerInfo 
from nolearn.lasagne.base import BatchIterator
from nolearn.lasagne.util import is_conv2d
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import uniform as sp_rand
from scipy.stats import randint
from sklearn.grid_search import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import skimage

#my modules
import my_net
from my_net import SaveBestWeights, AdjustVariable, EarlyStopping
import FlipBatchIterator as flip
import utils
from neural_data_setup import NeuralDataLoad
from timer import Timer


class Experiment(object):
    def __init__(self, flip=True, dropout=False, adaptive=True,
                    preloaded_path=None, pipeline=False, 
                    print_out=True, preloaded=False,  
                    log=True, plot_loss=True, plot=True,epochs=3000,  
                    roof_type=None, non_roofs=2, selective_data=None,
                    net_name=None, num_layers=None, max_roofs=None):
        '''
        Parameters:
        ------------
        flip: boolean
            whether we should perform data augmentation
        dropout: boolean
            whether we should perform dropout
        adaptive: boolean
            Whether learning rate and momentum should adapt over time
        '''
        self.pipeline = pipeline
       
        #if we are doing the pipeline, we already have a good name for the network, no need to add more info
        if self.pipeline:
            self.net_name = net_name
            #unpickle the scaler
            path = utils.get_path(params=True, neural_weights=True)
            with open(path+self.net_name+'_scaler.pickle') as f:
                self.scaler=pickle.load(f)
        else:
            #Load data
            print 'Loading data...\n'
            self.X, self.y = NeuralDataLoad(data_path=selective_data).load_data(roof_type=roof_type, non_roofs=non_roofs) 
            print 'Data is loaded \n'
            #set up the data scaler
            self.scaler = DataScaler()
            self.X = self.scaler.fit_transform(self.X)
            print self.X.shape
     
            #count the number of each type of class, add it to the network name
            if roof_type == 'Both':
                nonroof_num, metal_num, thatch_num = np.bincount(self.y)
            elif roof_type == 'metal':
                nonroof_num, metal_num = np.bincount(self.y)
                thatch_num = 0
            elif roof_type == 'thatch':
                nonroof_num, thatch_num = np.bincount(self.y)
                metal_num = 0
            else:
                raise ValueError('You have given an unknown roof_type to the network')
            self.net_name = 'conv{0}_{1}_metal{2}_thatch{3}_nonroof{4}'.format(num_layers, net_name, metal_num, thatch_num, nonroof_num)
            self.roof_type = roof_type

            #pickle the scaler so we can reuse it later
            path = utils.get_path(params=True, neural_weights=True) 
            with open('{0}{1}_scaler.pickle'.format(path, self.net_name), 'wb') as f:
                pickle.dump(self.scaler, f, -1)
 

        self.num_layers = num_layers
        print 'Final network name is: {0}'.format(self.net_name)
        self.flip = flip
        self.dropout = dropout
        self.non_roofs = non_roofs    #the proportion of non_roofs relative to roofs to be used in data

        self.epochs = epochs
        self.log = log
        self.plot_loss = True

        self.out_file = utils.get_path(in_or_out=utils.OUT, neural=True, data_fold=utils.TRAINING)+self.net_name
        print 'Setting up the Neural Net \n'
        self.adaptive_learning = adaptive
        self.setup_net(print_out=print_out)

        #preload weights if a path to weights was provided
        self.preloaded_path = preloaded_path 
        if preloaded_path is not None:
            preloaded_path = preloaded_path if preloaded_path.endswith('.pickle') else preloaded_path+'.pickle'
            self.preloaded_path = utils.get_path(neural_weights=True, params=True)+preloaded_path
            self.net.load_params_from(self.preloaded_path)      
        pdb.set_trace()



    def setup_net(self, print_out=True):
        if print_out:
            self.printer = PrintLogSave(out_file=self.out_file)
            on_epoch_finished = [self.printer, SaveBestWeights(), EarlyStopping(patience=200, out_file=self.out_file)]
            on_training_started = [SaveLayerInfo(out_file=self.out_file)]
        else:
            on_epoch_finished = [EarlyStopping(patience=200, out_file=self.out_file)]
            on_training_started = []

        if self.flip:
            #batch_iterator_train=flip.ResizeBatchIterator(batch_size=128) 
            batch_iterator_train=flip.FlipBatchIterator(batch_size=128)
        else:
            batch_iterator_train=flip.ResizeBatchIterator(batch_size=128) 

        if self.adaptive_learning:
            update_learning_rate = theano.shared(utils.float32(0.03))
            update_momentum = theano.shared(utils.float32(0.9))
            on_epoch_finished.append(AdjustVariable('update_learning_rate', start=0.03, stop=0.0001))
            on_epoch_finished.append(AdjustVariable('update_momentum', start=0.9, stop=0.999))
        else: 
            update_learning_rate = 0.01
            update_momentum = 0.9


        layers, layer_params = my_net.MyNeuralNet.produce_layers(self.num_layers, dropout=self.dropout)      
        self.net = my_net.MyNeuralNet(
            layers=layers,
            num_layers=self.num_layers,
            input_shape=(None, 3, utils.CROP_SIZE, utils.CROP_SIZE),
            output_num_units=3,

            output_nonlinearity=lasagne.nonlinearities.softmax,
            preproc_scaler = None, 
        
            #learning rates
            update_learning_rate=update_learning_rate,
            update_momentum=update_momentum,
        
            #printing
            net_name=self.net_name,
            on_epoch_finished=on_epoch_finished,
            on_training_started=on_training_started,

            #data augmentation
            batch_iterator_test= flip.ResizeBatchIterator(batch_size=128),
            batch_iterator_train=batch_iterator_train,

        
            max_epochs=self.epochs,
            verbose=1,
            **layer_params
            )
        return layer_params


    def train_test(self, fname_pretrain=None):
        '''
        Train and test neural network. Also print out evaluation.
        '''
        #pretraining
        if fname_pretrain:
            with open(fname_pretrain, 'rb') as f:
                net_pretrain = pickle.load(f)
        else:
            net_pretrain = None
        if net_pretrain is not None:
            self.load_params_from(net_pretrain)

        log = 'best_valid_loss\tbest_epoch\tbest_valid_accuracy\tlayers\tdropout\taugment\troof_type\ttime\tnet_name\n'
        self.printer.log_to_file(self.net, log, overwrite=True)

        #fitting the network to X_train
        with Timer() as t:
            self.net.fit(self.X, self.y)

        self.save_params_to_file(timer=t)
        self.net.save_weights()



    def save_params_to_file(self, timer=None):
        log = []
        log.append('{}'.format(self.num_layers))
        log.append('{}'.format(self.dropout))
        log.append('{}'.format(self.flip))
        log.append('{}'.format(self.roof_type))
        log.append('{}'.format(timer.secs))
        log.append('{}'.format(self.net_name))
        self.printer.log_to_file(self.net, '\t'.join(log))


    def test(self, X_test): 
        #apply the same scaling we applied to the training set
        X_test_scaled = self.scaler.transform2(X_test)

        #resize the patches from patch size to crop size
        X_test_cropped = np.empty((X_test_scaled.shape[0], X_test_scaled.shape[1], utils.CROP_SIZE, utils.CROP_SIZE))
        for i, x in enumerate(X_test_scaled):
            X_test_cropped[i, :, :, :] = utils.resize_neural_patch(x)
        return self.net.predict(X_test_cropped)


    def test_preloaded(self, plot_loss=True, test_case=None):  
        '''Preload weights, classify roofs and write evaluation
        To classify a single instance it must be passed in as the test_case parameter. Otherwise,
        the method will test the network on the test portion of the training set
        '''
        #save settings to file
        self.printer.log_to_file(self.net, self.__str__(), overwrite=True)
    
        #find predictions for test set
        #need to reduce it from 40 pixels down to 32 pixels
        min = (utils.PATCH_H - utils.CROP_SIZE)/2
        print self.X_test.shape
        self.X_test = self.X_test[:, :, min:min+utils.CROP_SIZE, min:min+utils.CROP_SIZE]
        print self.X_test.shape
        predicted = self.net.predict(self.X_test)
        self.evaluation(predicted, self.X_train, self.X_test, self.y_train, self.y_test)


    def evaluation(self, predicted, X_train, X_test, y_train, y_test):
        #print evaluation
        self.printer.log_to_file(self.net, confusion_matrix(y_test, predicted), binary=True, title='\n\nConfusion Matrix\n')
        self.printer.log_to_file(self.net, classification_report(y_test, predicted), title='\n\nReport\n')
    
        #save a plot of the validation and training losses
        #if self.plot_loss:
        self.net.save_loss()
       

    @staticmethod
    def report_grid(grid_scores, n_top=3):
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i , score in enumerate(top_scores):
            print "Model with rank: {0}".format(i+1)
            print "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    score.mean_validation_score,
                    np.std(score.cv_validation_scores))
            print "Parameters: {0}".format(score.parameters)
            print ""


    def optimize_params(self):
        params_grid = {"update_learning_rate": sp_rand(0.001,0.01), "momentum":sp_rand(0.9,2.0), "epochs":randint(50,300)}
        rsearch = RandomizedSearchCV(estimator=self.net, param_distributions=params_grid, n_iter=15, n_jobs=1)
        X, _, y, _ = self.roof_loader.load(max_roofs=100, test_percent=0, non_roofs=self.non_roofs)
        rsearch.fit(X,y)
        Experiment.report_grid(rsearch.grid_scores_)


    def __str__(self):
        out_list = list()
        for key, value in self.__dict__.items():
            out_list.append(str(key)+': '+str(value))
        out_list.append('\n')
        return '\n'.join(out_list)          


class DataScaler(StandardScaler):
    #Subclass of sklearn.StandardScaler that reshapes data as needed and then calls super to do scaling
    def fit_transform(self, X):
        X_shape = X.shape
        X = X.reshape(X_shape[0], X_shape[1]*X_shape[2]*X_shape[3])
        X = super(DataScaler, self).fit_transform(X)
        return X.reshape(X_shape[0], X_shape[1], X_shape[2], X_shape[3])

    def transform2(self, X):
        X_shape = X.shape
        #reshape for transformation
        if len(X.shape) == 3:
            single_image = True
            X = X[None, :,:,:] 
            X = X.reshape(1, X_shape[0]*X_shape[1]*X_shape[2])
        else:
            single_image = False
            X = X.reshape(X_shape[0], X_shape[1]*X_shape[2]*X_shape[3])
    
        X = super(DataScaler, self).transform(X)
    
        if single_image:
            X = np.squeeze(X)
            return X.reshape(X_shape[0], X_shape[1], X_shape[2])
        else:
            return X.reshape(X_shape[0], X_shape[1], X_shape[2], X_shape[3])

    def inverse_transform(self, X): 
        X_shape = X.shape
        X = X.reshape(X_shape[0], X_shape[1]*X_shape[2]*X_shape[3])
        X = super(DataScaler, self).inverse_transform(X)
        return X.reshape(X_shape[0], X_shape[1], X_shape[2], X_shape[3])


class PrintLogSave(PrintLog):
    def __init__(self, out_file=None):
        self.out_file = out_file
        with open(self.out_file+'_history', 'w') as f:
            f.write('epoch\ttrain_loss\tvalid_loss\ttrain_over_valid_loss_ratio\tvalid_accuracy\n')

    def __call__(self, nn, train_history): 
        with open(self.out_file+'_history', 'a') as f:
            f.write(self.table(nn, train_history))

    def table(self, nn, train_history):
        info = train_history[-1]
        return str(info['epoch'])+'\t'+ str(info['train_loss'])+ '\t'+str(info['valid_loss'])+'\t'+str( info['train_loss'] / info['valid_loss'])+'\t'+str(info['valid_accuracy'])+'\n'
        
    def log_to_file(self, nn, log, overwrite=False, binary=False, title=''):
        write_type = 'w' if overwrite else 'a'
        with open(self.out_file, write_type) as f:
            if binary:
                print >> file, log
            else:
                f.write(title)
                f.write(log)


class SaveLayerInfo(PrintLayerInfo):
    def __init__(self, out_file=None):
        self.out_file = out_file

    def __call__(self, nn, train_history):
        with open(self.out_file+'_layers', 'w') as f:
            f.write("## Layer information")
            f.write("\n")

            layers_contain_conv2d = is_conv2d(list(nn.layers_.values()))
            if not layers_contain_conv2d or (nn.verbose < 2):
                layer_info = self._get_layer_info_plain(nn)
                legend = None
            else:
                layer_info, legend = self._get_layer_info_conv(nn)
            f.write(layer_info)
            if legend is not None:
                file.write(legend)
            f.write(" \n")




def get_neural_training_params_from_file(file_name):
    parameters = dict()
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for par in reader:
            if len(par) == 2: 
                key = par[0].strip()
                if key == 'selective_data' or key == 'roof_type':
                    parameters[key] = (par[1].strip())
                else:
                    parameters[key] = int(float(par[1].strip()))
    return parameters


def get_parameter_file_and_ensemble_number():
    param_file = 10001
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:e:")
    except getopt.GetoptError:
        sys.exit(2)
        print 'Command line failed'
    for opt, arg in opts:
        if opt == '-f':
            param_file = arg
        if opt == '-e':
            ensemble_num = arg
    return param_file, ensemble_num


if __name__ == '__main__':
    param_file_num, ensemble_num = get_parameter_file_and_ensemble_number()

    param_file = 'params{}.csv'.format(param_file_num)
    params_path = '{0}{1}'.format(utils.get_path(params=True, neural=True), param_file)
    params = get_neural_training_params_from_file(params_path) 

    params['net_name'] = '{}_flip{}_dropout{}_adapt{}_ensemble{}'.format(params['selective_data'], params['flip'], params['dropout'], params['adaptive'], ensemble_num)
    print 'Network name is: {0}'.format(params['net_name'])

    experiment = Experiment(print_out=True, **params)

    to_do = 't'
    if to_do == 'o':
        experiment.optimize_params()
    elif to_do == 't':
        experiment.train_test() 
    else:
        pass

