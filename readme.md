# robot hand state prediction
Xi Liu<br>
This project is done for machine learning course at New York University in 2022 fall.<br>
State prediction given RGBD (RGB + Depth) images. The input is RGBD images of top view robot hand, after the use of several supervised learning algorithms, the output is vertex positions of each finger in meters.<br>
Each sample is made of three images from three different views. A custom dataset class is defined for lazy loading to deal with out of memory issue.

## method
### data preprocessing
A custom data processing class ```Data_Preprocessing``` is implemented in ```CNN_model.py```
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
combine ```ready_img_train``` and ```y_array``` into one array, and save that data persistently as a ```.joblib``` file using ```joblib.dump()```
```python
train_img0 = [ready_img_train, y_array]
dump(train_img0, 'preprocessed_testX.joblib')
```

### train model
start to train the model by calling ```cnn_model.main()```, many residual neural network architectures are used, including ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152.
```python
cnn_model = cnn_model.main(loadname = 'lx_preprocessed_data0.joblib', pre_trained_model = None)
```

### experimental results
image 0, 1, 2 represent 3 images of 3 different views in each sample<br>
Using ResNet50 with image 0 received a root mean square error score of 0.00863.<br>
Using ResNet50 with image 0, 1, 2 combined and using image 0 as test x received a root mean square error score of 57.29925.<br>
Using ResNet50 with image 0, 1, 2 combined and using image 1 as test x received a root mean square error score of 56.26985.<br>
Using ResNet50 with image 0, 1, 2 combined and using image 2 as test x received a root mean square error score of 63.34214.<br>
Using ResNet50 with image 1 received a root mean square error score of 65.07187.<br>
Using ResNet50 with image 2 received a root mean square error score of 65.48362.<br>
Using ResNet101 with image 0 received a root mean square error score of 65.96393.<br>
Using ResNet18 with image 0 received a root mean square error score of 66.22581.<br>
Using ResNet50 with image 0 received a root mean square error score of 66.72329.<br>

### discussion


### future work
It seems there are a lot of performance penalty due to the implementation of the language. For example, it took a lot of time to preprocess the data and traverse through the dataset. It would be a lot faster if the data is organized better and store in memory in a way that have better spatial and temporal locality. In the future, it seems C++ CUDA would be a better choice not only for preprocessing but also for training. Using the current approach, some time there would be run out of memory problem if saving the entire dataset as a malloced array in heap at once. In the future, load only when the training loop started to use the portion of the data.
