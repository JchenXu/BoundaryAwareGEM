import os
import sys
import numpy as np
import util
import h5py
import pickle
from plyfile import PlyData, PlyElement


def remove_unano(scene_data, scene_label, scene_data_id):
    keep_idx = np.where((scene_label > 0) & (scene_label < 41))  # 0: unanotated
    scene_data_clean = scene_data[keep_idx]
    scene_label_clean = scene_label[keep_idx]
    scene_data_id_clean = scene_data_id[keep_idx]
    return scene_data_clean, scene_label_clean, scene_data_id_clean


test_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


def gen_label_map():
    label_map = np.zeros(41)
    for i in range(41):
        if i in test_class:
            label_map[i] = test_class.index(i)
        else:
            label_map[i] = 0
    print(label_map)
    return label_map


def gen_pickle(split="val", root="DataSet/Scannet_v2"):
    if split == 'test':
        root = root + "/scans_test"
    else:
        root = root + "/scans"
    file_list = "scannetv2_%s.txt" % (split)
    with open(file_list) as fl:
        scene_id = fl.read().splitlines()

    scene_data = []
    scene_data_labels = []
    scene_data_id = []
    scene_data_num = []
    label_map = gen_label_map()
    for i in range(len(scene_id)):  # len(scene_id)
        print('process...', i)
        scene_namergb = os.path.join(root, scene_id[i], scene_id[i] + '_vh_clean_2.ply')
        scene_xyzlabelrgb = PlyData.read(scene_namergb)
        scene_vertex_rgb = scene_xyzlabelrgb['vertex']
        scene_data_tmp = np.stack((scene_vertex_rgb['x'], scene_vertex_rgb['y'],
                                   scene_vertex_rgb['z'], scene_vertex_rgb['red'],
                                   scene_vertex_rgb['green'], scene_vertex_rgb['blue']), axis=-1).astype(np.float32)
        scene_points_num = scene_data_tmp.shape[0]
        scene_point_id = np.array([c for c in range(scene_points_num)])
#        if split != 'test':
        scene_name = os.path.join(root, scene_id[i], scene_id[i] + '_vh_clean_2.labels.ply')
        scene_xyzlabel = PlyData.read(scene_name)
        scene_vertex = scene_xyzlabel['vertex']
        scene_data_label_tmp = scene_vertex['label']
        scene_data_tmp, scene_data_label_tmp, scene_point_id_tmp = remove_unano(scene_data_tmp,
                                                                                    scene_data_label_tmp,
                                                                                    scene_point_id)
#        else:
 #           scene_data_label_tmp = np.zeros((scene_data_tmp.shape[0])).astype(np.int32)
  #          scene_point_id_tmp = scene_point_id
        scene_data_label_tmp = label_map[scene_data_label_tmp]
        scene_data.append(scene_data_tmp)
        scene_data_labels.append(scene_data_label_tmp)
        scene_data_id.append(scene_point_id_tmp)
        scene_data_num.append(scene_points_num)

    pickle_out = open("scannet_%s_rgb21c_pointid.pickle" % (split), "wb")
    pickle.dump(scene_data, pickle_out, protocol=0)
    pickle.dump(scene_data_labels, pickle_out, protocol=0)
    pickle.dump(scene_data_id, pickle_out, protocol=0)
    pickle.dump(scene_data_num, pickle_out, protocol=0)
    pickle_out.close()


if __name__ == '__main__':
    root = "/home/xxx/pc_data/scannet/scannet_v2"  # modify this path to your Scannet v2 dataset Path
#    gen_pickle(split='train', root=root)
#    gen_pickle(split='val', root=root)
    gen_pickle(split='test', root=root)

    print('Done!!!')
