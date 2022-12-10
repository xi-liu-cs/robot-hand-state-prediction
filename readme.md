# robot hand state prediction
Xi Liu<br>
This project is done for machine learning course at New York University in 2022 fall.<br>
State prediction given RGBD (RGB + Depth) images. The input is RGBD images of top view robot hand, after the use of several supervised learning algorithms, the output is vertex positions of each finger in meters.<br>
Each sample is made of three images from three different views. A custom dataset class is defined for lazy loading to deal with out of memory issue.

## method
### data preprocessing
A custem data processing class ```Data_Preprocessing``` is implemented in ```CNN_model.py```
```python
dp = CNN_model.Data_Preprocessing()
```
load ```data_train``` and ```data_test``` using custom data loader ```load_images()```
```python
data_train = CNN_model.load_images(path = './lazydata/', isTrain = True)
data_test = CNN_model.load_images(path = './lazydata/', isTrain = False)
```
convert ```data_train``` and ```data_test``` from tensor to array using ```tensorToArray()```
```python
img0_array_test, img1_array_test, img2_array_test, depth_array_test, field_id_array = dp.tensorToArray(data = data_test, isTrain = False)
img0_array_train, img1_array_train, img2_array_train, depth_array_train, y_array = dp.tensorToArray(data = data_train, isTrain = True)
```
normalize depth arrays and image arrays using ```depth_normalization(), img_normalization(), combine_image_depth()```
```python
normalized_depth_train = dp.depth_normalization(depth = depth_array_train)
normalized_img0_train = dp.img_normalization(img = img0_array_train)
new_img_train = dp.combine_image_depth(img = normalized_img0_train, depth = normalized_depth_train, whichImg = 0)
ready_img_train = dp.reshape_data(new_img_train)
```
