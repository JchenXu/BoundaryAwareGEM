"""
Modified from PointNet++: https://github.com/charlesq34/pointnet2
Author: Wenxuan Wu
Date: July 2018
"""
import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'scannet'))
sys.path.append('/home/gongjingyu/gcode/RGBD/data/S3DIS')
import provider
import tf_util
import scannet_dataset_rgb
import time
import util
import pointconv_util

colors = util.create_color_palette()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=1501, help='Epoch to run [default: 501]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 6]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
EPOCH_CNT_WHOLE = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

BANDWIDTH = 0.05

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
Point_Util = os.path.join(BASE_DIR, 'utils', 'pointconv_util.py')
LOG_DIR = FLAGS.log_dir + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (Point_Util, LOG_DIR))
os.system('cp %s %s' % ('PointConv.py', LOG_DIR))
os.system('cp train_S3DIS_IoU.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
L2_LOSS_WEIGHT = 0.0

HOSTNAME = socket.gethostname()

NUM_CLASSES = 13

# Shapenet official train/test split
#DATA_PATH = os.path.join(BASE_DIR, 'scannet')
#print("start loading training data ...")
#TRAIN_DATASET = scannet_dataset_rgb.ScannetDataset(root=DATA_PATH, block_points=NUM_POINT, split='train', with_rgb=True)
#print("start loading validation data ...")
#TEST_DATASET = scannet_dataset_rgb.ScannetDataset(root=DATA_PATH, block_points=NUM_POINT, split='val', with_rgb=True)
# print("start loading whole scene validation data ...")
# TEST_DATASET_WHOLE_SCENE = scannet_dataset_rgb.ScannetDatasetWholeScene(root=DATA_PATH, block_points=NUM_POINT, split='val')

#load S3DIS dataset
"""
ALL_FILES = provider.getDataFiles('indoor3d_sem_seg_hdf5_data_v7/all_files.txt')
room_filelist = [line.rstrip() for line in open('indoor3d_sem_seg_hdf5_data_v7/room_filelist.txt')]
"""

# Load ALL data
"""
data_batch_list = []
label_batch_list = []
for h5_filename in ALL_FILES:
    data_batch, label_batch = provider.loadDataFile(h5_filename)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
data_batches = data_batches[:, :, :6]
print(data_batches.shape)
print(label_batches.shape)

test_area = 'Area_'+str(FLAGS.test_area)
train_idxs = []
test_idxs = []
for i,room_name in enumerate(room_filelist):
    if test_area in room_name:
        test_idxs.append(i)
    else:
        train_idxs.append(i)

train_data = data_batches[train_idxs,...]
train_label = label_batches[train_idxs]
test_data = data_batches[test_idxs,...]
test_label = label_batches[test_idxs]
print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)
num_test_data = int(test_data.shape[0] // 4)
"""
class MyData():
    def __init__(self, sample_num, path):
        if path[-4:] == 'data':
            self.shape = np.array([sample_num, 4096, 6])
        else:
            self.shape = np.array([sample_num, 4096])
        self.sample_list = []
        for i in range(sample_num):
            self.sample_list.append(path+"/"+"%05d"%i+".npy")
    def __getitem__(self, index):
        return np.load(self.sample_list[index])
    def __len__(self):
        return len(self.sample_list)
train_data = MyData(51182, '/all-data/sv3-disk2/gongjingyu_home/gcode/RGBD/dataset/S3DIS/indoor3d_sem_seg_hdf5_data_v7_npy/train_data')
train_label = MyData(51182, '/all-data/sv3-disk2/gongjingyu_home/gcode/RGBD/dataset/S3DIS/indoor3d_sem_seg_hdf5_data_v7_npy/train_label')
test_data = MyData(8334, '/all-data/sv3-disk2/gongjingyu_home/gcode/RGBD/dataset/S3DIS/indoor3d_sem_seg_hdf5_data_v7_npy/test_data')
test_label = MyData(8334, '/all-data/sv3-disk2/gongjingyu_home/gcode/RGBD/dataset/S3DIS/indoor3d_sem_seg_hdf5_data_v7_npy/test_label')
num_test_data = int(test_data.shape[0] // 4)
class_sample_number = [13415141, 12095149, 10941972,  1258616,   963770,   810160, 2760541,  1961664,  2613820,   280297,  2112759,  436227, 6372578]
class_sample_number = np.array(class_sample_number).astype(np.float32)
class_sample_weight = class_sample_number / np.sum(class_sample_number)
class_sample_weight = 1 / np.log(1.15+class_sample_weight)
print("class sample weight")
print(class_sample_weight)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss
            boundary_label, boundary_loss = MODEL.get_boundary_model_loss(labels_pl, pointclouds_pl, is_training_pl, NUM_CLASSES, BANDWIDTH, bn_decay=bn_decay, weight_decay = L2_LOSS_WEIGHT)
            pred, end_points = MODEL.get_model(boundary_label, pointclouds_pl, is_training_pl, NUM_CLASSES, BANDWIDTH, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, smpws_pl, boundary_loss)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=50)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
        whole_test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'whole_scene'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        #sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points,
               'boundary_loss': boundary_loss}

        best_mIoU = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            start_time = time.time()
            train_one_epoch(sess, ops, train_writer, epoch)
            end_time = time.time()
            log_string('one epoch time: %.4f'%(end_time - start_time))
            eval_mIoU = eval_one_epoch(sess, ops, test_writer)
            if eval_mIoU > best_mIoU:
                best_mIoU = eval_mIoU
            if eval_mIoU >= 0.565:
                eval_all_mIoU = eval_all_one_epoch(sess, ops, test_writer)
                print("test result mIoU %.4f"%eval_all_mIoU)
                if eval_all_mIoU >= 0.58:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                    log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw

        dropout_ratio = np.random.random()*0.875 # 0-0.875
        drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
        batch_data[i,drop_idx,:] = batch_data[i,0,:]
        batch_label[i,drop_idx] = batch_label[i,0]
        batch_smpw[i,drop_idx] *= 0
    return batch_data, batch_label, batch_smpw

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw
    return batch_data, batch_label, batch_smpw

def my_get_batch(dataset, labelset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    for i in range(bsize):
        ps = dataset[idxs[i+start_idx]]
        seg = labelset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
    batch_smpw = class_sample_weight[batch_label]
    return batch_data, batch_label, batch_smpw

def train_one_epoch(sess, ops, train_writer, epoch=None):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    #current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label)
    #current_data = current_data[:2*num_test_data,...]
    #current_label = current_label[:2*num_test_data,...]
    train_idxs = np.arange(0, len(train_data))
    np.random.shuffle(train_idxs)
    train_idxs = train_idxs[:2*num_test_data]
    # Shuffle train samples
    #train_idxs = np.arange(0, len(TRAIN_DATASET))
    #train_idxs = np.arange(0, train_data.shape[0])
    #np.random.shuffle(train_idxs)
    #num_batches = int(current_data.shape[0]/BATCH_SIZE)
    num_batches = int(train_idxs.shape[0]/BATCH_SIZE)

    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_iou_deno = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        #batch_data, batch_label, batch_smpw = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        #batch_data, batch_label, batch_smpw = get_batch(train_data, train_idxs, start_idx, end_idx)
        #batch_data = current_data[start_idx:end_idx,...]
        #batch_label = current_label[start_idx:end_idx,...]
        batch_data, batch_label, batch_smpw_tmp = my_get_batch(train_data, train_label, train_idxs, start_idx, end_idx)
        #batch_smpw = np.ones(batch_label.shape)
        batch_smpw = (batch_data[:,:,0] >= -0.25) & (batch_data[:,:,0] <= 0.25) & (batch_data[:,:,1] >= -0.25) & (batch_data[:,:,1] <= 0.25)
        batch_smpw = batch_smpw.astype(np.float32)
        batch_smpw = batch_smpw * batch_smpw_tmp
        # Augment batched point clouds by rotation
        batch_data[:,:,:3] = provider.rotate_point_cloud_z(batch_data[:,:,:3])
        #aug_data = provider.rotate_point_cloud(batch_data)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                    ops['labels_pl']: batch_label,
                    ops['smpws_pl']:batch_smpw,
                    ops['is_training_pl']: is_training,}

        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        iou_deno = 0
        for l in range(NUM_CLASSES):
            iou_deno += np.sum((pred_val==l) | (batch_label==l))
        total_iou_deno += iou_deno
        loss_sum += loss_val
        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            log_string('total IoU: %f' % (total_correct / float(total_iou_deno)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            total_iou_deno = 0

# evaluate on randomly chopped scenes
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(test_data))
    np.random.shuffle(test_idxs)
    test_idxs = test_idxs[:num_test_data]
    num_batches = int(test_idxs.shape[0]/BATCH_SIZE)
    #test_idxs = np.arange(0, len(TEST_DATASET))
    #current_data = test_data[:,0:NUM_POINT,:]
    #current_label = np.squeeze(test_label)
    #current_data, current_label, _ = provider.shuffle_data(test_data[:,0:NUM_POINT,:], test_label)
    #current_data = current_data[:num_test_data,...]
    #current_label = current_label[:num_test_data,...]
    #num_batches = int(current_data.shape[0]/BATCH_SIZE)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    labelweights = np.zeros(13)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        #batch_data, batch_label, batch_smpw = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)
        #batch_data = current_data[start_idx:end_idx,...]
        #batch_label = current_label[start_idx:end_idx,...]
        batch_data, batch_label, _ = my_get_batch(test_data, test_label, test_idxs, start_idx, end_idx)
        #batch_smpw = np.ones(batch_label.shape)
        batch_smpw = (batch_data[:,:,0] >= -0.25) & (batch_data[:,:,0] <= 0.25) & (batch_data[:,:,1] >= -0.25) & (batch_data[:,:,1] <= 0.25)
        batch_smpw = batch_smpw.astype(np.float32)

        batch_data[:,:, :3] = provider.rotate_point_cloud_z(batch_data[:, :, :3])
        #aug_data = provider.rotate_point_cloud(batch_data)
        bandwidth = BANDWIDTH

        feed_dict = {ops['pointclouds_pl']: batch_data,
                    ops['labels_pl']: batch_label,
                    ops['smpws_pl']: batch_smpw,
                    ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val, boundary_loss = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred'], ops['boundary_loss']], feed_dict=feed_dict)

        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2) # BxN
        correct = np.sum((pred_val == batch_label) & (batch_smpw>0)) # evaluate only on 20 categories but not unknown
        total_correct += correct
        total_seen += np.sum((batch_smpw>0))
        loss_sum += loss_val
        tmp,_ = np.histogram(batch_label,range(14))
        labelweights += tmp
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((batch_label==l) & (batch_smpw>0))
            total_correct_class[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))
            total_iou_deno_class[l] += np.sum(((pred_val==l) | (batch_label==l)) & (batch_smpw>0))

    mIoU = np.mean(np.array(total_correct_class[:])/(np.array(total_iou_deno_class[:],dtype=np.float)+1e-8))
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval point avg class IoU: %f' % (mIoU))
    log_string('eval point accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval point avg class acc: %f' % (np.mean(np.array(total_correct_class[:])/(np.array(total_seen_class[:],dtype=np.float)+1e-8))))
    iou_per_class_str = '------- IoU --------\n'
    for l in range(NUM_CLASSES):
        iou_per_class_str += 'class %d, acc: %f \n' % (l,total_correct_class[l]/float(total_iou_deno_class[l]))
    log_string(iou_per_class_str)
    EPOCH_CNT += 1
    return mIoU

# evaluate on randomly chopped scenes
def eval_all_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(test_data))
    num_batches = int(test_idxs.shape[0]/BATCH_SIZE)
    #test_idxs = np.arange(0, len(TEST_DATASET))
    #current_data = test_data[:,0:NUM_POINT,:]
    #current_label = np.squeeze(test_label)
    #num_batches = int(current_data.shape[0]/BATCH_SIZE)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d ALL EVALUATION ----'%(EPOCH_CNT))

    labelweights = np.zeros(13)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        #batch_data, batch_label, batch_smpw = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)
        #batch_data = current_data[start_idx:end_idx,...]
        #batch_label = current_label[start_idx:end_idx,...]
        batch_data, batch_label, _ = my_get_batch(test_data, test_label, test_idxs, start_idx, end_idx)
        #batch_smpw = np.ones(batch_label.shape)
        batch_smpw = (batch_data[:,:,0] >= -0.25) & (batch_data[:,:,0] <= 0.25) & (batch_data[:,:,1] >= -0.25) & (batch_data[:,:,1] <= 0.25)
        batch_smpw = batch_smpw.astype(np.float32)

        batch_data[:,:, :3] = provider.rotate_point_cloud_z(batch_data[:, :, :3])
        #aug_data = provider.rotate_point_cloud(batch_data)
        bandwidth = BANDWIDTH

        feed_dict = {ops['pointclouds_pl']: batch_data,
                    ops['labels_pl']: batch_label,
                    ops['smpws_pl']: batch_smpw,
                    ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val, boundary_loss = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred'], ops['boundary_loss']], feed_dict=feed_dict)

        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2) # BxN
        correct = np.sum((pred_val == batch_label) & (batch_smpw>0)) # evaluate only on 20 categories but not unknown
        total_correct += correct
        total_seen += np.sum((batch_smpw>0))
        loss_sum += loss_val
        tmp,_ = np.histogram(batch_label,range(14))
        labelweights += tmp
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((batch_label==l) & (batch_smpw>0))
            total_correct_class[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))
            total_iou_deno_class[l] += np.sum(((pred_val==l) | (batch_label==l)) & (batch_smpw>0))

    mIoU = np.mean(np.array(total_correct_class[:])/(np.array(total_iou_deno_class[:],dtype=np.float)+1e-8))
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval point avg class IoU: %f' % (mIoU))
    log_string('eval point accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval point avg class acc: %f' % (np.mean(np.array(total_correct_class[:])/(np.array(total_seen_class[:],dtype=np.float)+1e-8))))
    iou_per_class_str = '------- IoU --------\n'
    for l in range(NUM_CLASSES):
        iou_per_class_str += 'class %d, acc: %f \n' % (l,total_correct_class[l]/float(total_iou_deno_class[l]))
    log_string(iou_per_class_str)
    EPOCH_CNT += 1
    return mIoU


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
