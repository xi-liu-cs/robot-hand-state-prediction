# robot hand state prediction
This project is state prediction given RGBD (RGB + Depth) images. The input is RGBD images of top view robot hand, after the use of several supervised learning algorithms, the output is vertex positions of each finger in meters.<br>
Each sample is made of three images from three different views. A custom dataset class is defined for lazy loading to deal with out of memory issue.
