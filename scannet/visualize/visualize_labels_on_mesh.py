# Example script to visualize labels in the evaluation format on the corresponding mesh.
# Inputs:
#   - predicted labels as a .txt file with one line per vertex
#   - the corresponding *_vh_clean_2.ply mesh
# Outputs a .ply with vertex colors, a different color per value in the predicted .txt file
#
# example usage: visualize_labels_on_mesh.py --pred_file [path to predicted labels file] --mesh_file [path to the *_vh_clean_2.ply mesh] --output_file [output file]

# python imports
import math
import os, sys, argparse
import inspect
import json

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)
try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import util
import util_3d

def visualize(pred_file, mesh_file, output_file):
    if not output_file.endswith('.ply'):
        util.print_error('output file must be a .ply file')
    colors = util.create_color_palette()
    num_colors = len(colors)
    ids = util_3d.load_ids(pred_file)
    with open(mesh_file, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        if num_verts != len(ids):
            util.print_error('#predicted labels = ' + str(len(ids)) + 'vs #mesh vertices = ' + str(num_verts))
        # *_vh_clean_2.ply has colors already
        for i in range(num_verts):
            if ids[i] >= num_colors:
                util.print_error('found predicted label ' + str(ids[i]) + ' not in nyu40 label set')
            color = colors[ids[i]]
            plydata['vertex']['red'][i] = color[0]
            plydata['vertex']['green'][i] = color[1]
            plydata['vertex']['blue'][i] = color[2]
    plydata.write(output_file)

def visualize_boundary(pred_file, mesh_file, output_file, output_file_possi=None):
    if not output_file.endswith('.ply'):
        util.print_error('output file must be a .ply file')
    colors = util.create_color_palette()
    num_colors = len(colors)
    ids, boundary, boundary_possi = util_3d.load_ids_boundary(pred_file)
    with open(mesh_file, 'rb') as f:
        plydata = PlyData.read(f)
        f.seek(0)
        plydata_possi = PlyData.read(f)

        num_verts = plydata['vertex'].count
        if num_verts != len(ids):
            util.print_error('#predicted labels = ' + str(len(ids)) + 'vs #mesh vertices = ' + str(num_verts))
        # *_vh_clean_2.ply has colors already
        for i in range(num_verts):
            if ids[i] >= num_colors:
                util.print_error('found predicted label ' + str(ids[i]) + ' not in nyu40 label set')
            if boundary[i] == 0:
                plydata['vertex']['red'][i] = 255
                plydata['vertex']['green'][i] = 0
                plydata['vertex']['blue'][i] = 0

        num_verts = plydata_possi['vertex'].count
        if num_verts != len(ids):
            util.print_error('#predicted labels = ' + str(len(ids)) + 'vs #mesh vertices = ' + str(num_verts))
        # *_vh_clean_2.ply has colors already
        for i in range(num_verts):
            if ids[i] >= num_colors:
                util.print_error('found predicted label ' + str(ids[i]) + ' not in nyu40 label set')
            plydata_possi['vertex']['red'][i] = int(255 * (1 - boundary_possi[i]))
            plydata_possi['vertex']['green'][i] = int(255 * (1 - boundary_possi[i]))
            plydata_possi['vertex']['blue'][i] = int(255 * (1 - boundary_possi[i]))

    plydata.write(output_file)
    plydata_possi.write(output_file_possi)
