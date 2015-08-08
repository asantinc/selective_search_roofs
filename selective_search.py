import tempfile
import subprocess
import shlex
import os
import sys
import pdb
import getopt
import numpy as np
import scipy.io

from reporting import Evaluation, Detections
import utils
from timer import Timer


script_dirname = os.path.abspath(os.path.dirname(__file__))


def get_windows(image_fnames, cmd='selective_search', k=200, scale=1.08):
    """
    Run MATLAB Selective Search code on the given image filenames to
    generate window proposals.

    Parameters
    ----------
    image_filenames: strings
        Paths to images to run on.
    cmd: string
        selective search function to call:
            - 'selective_search' for a few quick proposals
            - 'selective_seach_rcnn' for R-CNN configuration for more coverage.
    """
    # Form the MATLAB script command that processes images and write to
    # temporary results file.
    f, output_filename = tempfile.mkstemp(suffix='.mat')
    os.close(f)
    fnames_cell = '{' + ','.join("'{}'".format(x) for x in image_fnames) + '}'
    command = "{}({}, '{}', {}, {})".format(cmd, fnames_cell, output_filename, k, scale)
    print(command)

    # Execute command in MATLAB.
    mc = "matlab -nojvm -r \"try; {}; catch; exit; end; exit\"".format(command)
    pid = subprocess.Popen(
        shlex.split(mc), stdout=open('/dev/null', 'w'), cwd=script_dirname)
    retcode = pid.wait()
    if retcode != 0:
        raise Exception("Matlab script did not exit successfully!")

    # Read the results and undo Matlab's 1-based indexing.
    all_boxes = list(scipy.io.loadmat(output_filename)['all_boxes'][0])
    subtractor = np.array((1, 1, 0, 0))[np.newaxis, :]
    all_boxes = [boxes - subtractor for boxes in all_boxes]

    # Remove temporary file, and return.
    os.remove(output_filename)
    if len(all_boxes) != len(image_fnames):
        raise Exception("Something went wrong computing the windows!")
    return all_boxes


def draw_windows(img_path, windows):
    windows = np.array(windows)
    img = cv2.imread(img_path)
    for win in windows:
        cv2.rectangle(img, (win[1], win[0]), (win[3], win[2]), (0,0,0), 2)
    return img

def selectionboxes2polygons(boxes):
    #the proposals are passed in as y1,x1, y2,x2

    polygons = np.empty((boxes.shape[0], 4, 2))    
    boxes = np.array(boxes)
    #follow the same convention as in order_points:
    #order points in
    #1.top_left 2.top_right 3.bottom_right 4.bottom_left order

    #1. top left point
    polygons[:,0,0] = boxes[:,1]#x1
    polygons[:,0,1] = boxes[:,0]#y1

    #3. bottom right
    polygons[:,2,0] = boxes[:,3]#x2
    polygons[:,2,1] = boxes[:,2]#y2

    #2. top right
    polygons[:,1,0] = boxes[:,3]# == x2
    polygons[:,1,1] = boxes[:,0]# == y1

    #4. bottom left
    polygons[:,3,0] = boxes[:,1] # == x1
    polygons[:,3,1] = boxes[:,2] # == y2

    return polygons

def get_parameters():
    k = 150
    scale = 0.8 
    try:
        opts, args = getopt.getopt(sys.argv[1:], "k:s:")
    except getopt.GetoptError:
        sys.exit(2)
        print 'Command line failed'
    for opt, arg in opts:
        if opt == '-k':
            k = int(float(arg))
        elif opt == '-s':
            scale = float(arg)
    return k, scale



def save_training_FP_and_TP_helper(img_name, detections, patches_path, general_path, img, roof_type, extraction_type, color):
    #this is where we write the detections we're extraction. One image per roof type
    #we save: 1. the patches and 2. the image with marks of what the detections are, along with the true roofs (for debugging)
    img_debug = np.copy(img) 

    if roof_type == 'background':
        utils.draw_detections(self.evaluation.correct_roofs['metal'][img_name], img_debug, color=(0, 0, 0), thickness=2)
        utils.draw_detections(self.evaluation.correct_roofs['thatch'][img_name], img_debug, color=(0, 0, 0), thickness=2)
    else:
        utils.draw_detections(self.evaluation.correct_roofs[roof_type][img_name], img_debug, color=(0, 0, 0), thickness=2)

    for i, detection in enumerate(detections):
        #extract the patch, rotate it to a horizontal orientation, save it
        bitmap = np.zeros((img.shape[:2]), dtype=np.uint8)
        padded_detection = utils.add_padding_polygon(detection, bitmap)
        warped_patch = utils.four_point_transform(img, padded_detection)
        cv2.imwrite('{0}{1}_{2}_roof{3}.jpg'.format(patches_path, roof_type, img_name[:-4], i), warped_patch)
        
        #mark where roofs where taken out from for debugging
        utils.draw_polygon(padded_detection, img_debug, fill=False, color=color, thickness=2, number=i)

    #write this type of extraction and the roofs to an image
    cv2.imwrite('{0}{1}_{2}_extract_{3}.jpg'.format(general_path, img_name[:-4], roof_type, extraction_type), img_debug)




def save_training_TP_FP_using_voc(self, evaluation, img_names, in_path, out_folder_name, neg_thresh=0.3):
    '''use the voc scores to decide if a patch should be saved as a TP or FP or not
    '''
    general_path = utils.get_path(neural=True, data_fold=utils.TRAINING, in_or_out=utils.IN, out_folder_name=out_folder_name)
    path_true = general_path+'truepos_from_selective_search/'
    utils.mkdir(path_true)

    path_false = general_path+'falsepos_from_selective_search/'
    utils.mkdir(path_false)

    for img_name in img_names:
        good_detections = defaultdict(list)
        bad_detections = defaultdict(list)
        try:
            if viola: #viola training will need grayscale patches
                img = cv2.imread(in_path+img_name, flags=cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.equalizeHist(img)
            else: #neural network will need RGB
                img = cv2.imread(in_path+img_name, flags=cv2.IMREAD_COLOR)
        except:
            print 'Cannot open image'
            sys.exit(-1)

        for roof_type in utils.ROOF_TYPES:
            detection_scores = detections.best_score_per_detection[img_name][roof_type]
            for detection, score in detection_scores:
                if score > 0.5:
                    #true positive
                    good_detections[roof_type].append(detection)
                if score < neg_thres:
                    #false positive
                    bad_detections[roof_type].append(detection)
                
        for roof_type in utils.ROOF_TYPES:
            extraction_type = 'good'
            save_training_FP_and_TP_helper(img_name, good_detections[roof_type], path_true, general_path, img, roof_type, extraction_type, (0,255,0))               
            extraction_type = 'background'
            save_training_FP_and_TP_helper(img_name, bad_detections[roof_type], path_false, general_path, img, roof_type, extraction_type, (0,0,255))               


def copy_images(data_fold):
    if data_fold == utils.TRAINING:
        prefix = 'training_'
    elif data_fold == utils.VALIDATION:
        prefix = 'validation_'

    in_path = utils.get_path(in_or_out = utils.IN, data_fold=data_fold)
    pdb.set_trace()
    for img_name in os.listdir(in_path):
        if img_name.endswith('jpg') or img_name.endswith('xml'):
            #move the image over and save it with a prefix       
            subprocess.check_call('cp {} {}'.format(in_path+img_name, prefix+img_name), shell=True)

if __name__ == '__main__':
    copy_images(utils.TRAINING)
    copy_images(utils.VALIDATION)
    '''
    img_names = [img for img in os.listdir(script_dirname) if img.endswith('jpg')]
    image_filenames = [script_dirname+'/'+img for img in os.listdir(script_dirname) if img.endswith('jpg') and img.startswith('training_')]

    #get the proposals
    k, scale = get_parameters()
    sim = '2'
    color = 'hsv'

    with Timer() as t:
        boxes = get_windows(image_filenames, k=k, scale=scale)

    detections = Detections()
    detections.total_time = t.secs
    folder_name = 'output_k{}_scale{}_sim{}_color{}/'.format(k, scale, sim, color)
    utils.mkdir(out_folder_path=folder_name)

    evaluation = Evaluation(use_corrected_roofs=True,
                        report_name='report.txt', method='windows', 
                        folder_name=folder_name,  out_path=folder_name, 
                        detections=detections, in_path=script_dirname+'/')
    
    #score the proposals
    for img, proposals in zip(img_names, boxes):
        print 'Evaluating {}'.format(img)
        print("Found {} windows".format(len(proposals)))

        proposals = selectionboxes2polygons(proposals)
        detections.set_detections(detection_list=proposals,roof_type='metal', img_name=img)  
        detections.set_detections(detection_list=proposals,roof_type='thatch', img_name=img)  
        print 'Evaluating...'
        evaluation.score_img(img, (1200,2000)) 

        evaluation.save_images(img)

    
    evaluation.print_report() 
    '''
