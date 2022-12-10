# robot hand state prediction
This project is state prediction given RGBD (RGB + Depth) images. The input is RGBD images of top view robot hand, after the use of several supervised learning algorithms, the output is vertex positions of each finger in meters.<br>
Each sample is made of three images from three different views. A custom dataset class is defined for lazy loading to deal with out of memory issue.

## method
### data preprocessing
1. tensor to array
```python
img0_array_test, img1_array_test, img2_array_test, depth_array_test, field_id_array = dp.tensorToArray(data = data_test, isTrain = False)
img0_array_train, img1_array_train, img2_array_train, depth_array_train, y_array = dp.tensorToArray(data = data_train, isTrain = True)
```
