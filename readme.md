# robot hand state prediction
This project is state prediction given RGBD (RGB + Depth) images. The input is RGBD images of top view robot hand, after the use of several supervised learning algorithms, the output is vertex positions of each fingers in meters.
Each sample is made of three images from three different views. A custom dataset class is defined for lazy loading to deal with out of memory issue, so that the relevant files are opened only when the getitem function is called.
