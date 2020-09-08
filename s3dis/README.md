## Semantic Segmentation of Indoor Scenes

### Dataset

Donwload prepared HDF5 data for training:

    sh download_data.sh

(optional) Download 3D indoor parsing dataset (<a href="http://buildingparser.stanford.edu/dataset.html">S3DIS Dataset</a>) for testing and visualization. Version 1.2 of the dataset is used in this work.


To prepare your own HDF5 data, you need to firstly download 3D indoor parsing dataset and then use `python collect_indoor3d_data.py` for data re-organization and `python gen_indoor3d_h5.py` to generate HDF5 files.

