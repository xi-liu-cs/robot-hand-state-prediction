# robot hand state prediction
Xi Liu<br>
This project is done for machine learning course at New York University in 2022 fall.<br>
State prediction given RGBD (RGB + Depth) images. The input is RGBD images of top view robot hand, after the use of several supervised learning algorithms, the output is vertex positions of each finger in meters.<br>
Each sample is made of three images from three different views. A custom dataset class is defined for lazy loading to deal with out of memory issue to load only when the training iteration started to use the portion of the data.

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
dump(train_img0, 'lx_preprocessed_data0.joblib')
```

### train model
start to train the model by calling ```cnn_model.main()```, many residual neural network architectures are used, including ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152.
```python
cnn_model = cnn_model.main(loadname = 'lx_preprocessed_data0.joblib', pre_trained_model = None)
model_scripted = torch.jit.script(cnn_model)
model_scripted.save('res50_pretrained_model.pt') # save model
sub = submission.Submission()
df = sub.submit(filename = 'preprocessed_testX.joblib', modelname = 'res50_pretrained_model.pt)
```
```python
class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
```
let $\mathcal{F}$ be the class of functions the the network architecture can satisfy 
```math
\displaylines
{
\forall f \in \mathcal{F}, \exists \text{ weights } w, \text{ biases } b\\
f_{\mathcal{F}} ^ * = arg\,min_f L(X, y, f) \text{ subject to } f \in \mathcal{F}
}
```

### experimental results
image 0, 1, 2 represent 3 images of 3 different views in each sample<br>
Using ResNet50 with image 0 with 20 epochs received a root mean square error score of 0.00707.<br>
Using ResNet50 with image 0 with 25 epochs received a root mean square error score of 0.01072.<br>
Using ResNet50 with image 0 with 22 epochs received a root mean square error score of 0.0149.<br>
Using ResNet50 with image 0 with 15 epochs received a root mean square error score of 0.01815.<br>
Using ResNet50 with image 2 with 20 epochs received a root mean square error score of 0.02464.<br>
Using ResNet50 with image 0 with 30 epochs received a root mean square error score of 0.0269.<br>
Using ResNet50 with image 0 with 40 epochs received a root mean square error score of 0.03257.<br>
Using ResNet50 with image 0, 1, 2 combined using image 0 as test x received a root mean square error score of 0.03367.<br>
Using ResNet50 with image 1 with 20 epochs received a root mean square error score of 0.03531.<br>


### discussion
Using ResNet50 with image 0 with 20 epochs received a root mean square error score of 0.00707. Unexpectedly, using image 0 performed better than using all of image 0, 1, 2 at the same time. Intuitively, one might think that using three images together would have a lower error since it learned from not only one view of the robot hand, but three different views, but it seems that optimization is not so straightforward.

### future work
It seems there are a lot of performance penalty in terms of time taken due to the implementation of the language and the Torch library. For example, it took a lot of time to preprocess the data and traverse through the dataset. It would be a lot faster if the data is organized better and stored in memory in a way that have better spatial and temporal locality. In the future, a combination of C++, CUDA, and GLSL would be a better choice for training and preprocessing, in which training the convolutional neural network would be done inside the fragment shader exploiting the parallelism of GPUs.
