# BrawlStars AI Project

## __Note that this project has been discontinued due to the difficulty of keeping up with a dynamic online game and the inefficiency of training without a simulator.__

[Blog Post 1](http://www.henrypan.com/blog/machine-learning/2019/04/20/Brawlstars-AI.html) and [Blog Post 2](http://www.henrypan.com/blog/reinforcement-learning/2019/04/25/Brawlstars-RL.html) recorded my journey.

## Demo:
![Demo](https://github.com/workofart/brawlstars-ai/raw/master/object_detection/demo/player_ally_enemy_v2.gif)


## Requirements:
- Android Simulator to run Brawlstars
- Brawlstars (This project is based on version 16.176)
- `Setup.py` contains all the dependencies necessary
- Pre-trained convolutional neural networks (https://github.com/tensorflow/models), specifically `ssd_mobilenet_v2`

# Notes

## Steps for Supervised Learning (on Windows)

### 1. Convert XML labels to CSV

Run in $REPO/object_detection
```
& "D:/Program Files/Anaconda3/envs/tf-gpu/python.exe" d:/Github/brawlstars-ai/object_detection/xml_to_csv.py
```

### 2. Generate TF_Record
**Don't forget to change the labels in the `generate_tfrecord.py` file**
Run in $REPO/object_detection
```
& "D:/Program Files/Anaconda3/envs/tf-gpu/python.exe" .\generate_tfrecord.py --csv_input=data/train.csv --output_path=data/train.record --image_dir=img/train
```

```
& "D:/Program Files/Anaconda3/envs/tf-gpu/python.exe" .\generate_tfrecord.py --csv_input=data/test.csv --output_path=data/test.record --image_dir=img/test
```

### 3. Train Object Detection Model
Run in $REPO/object_detection
**Don't forget to configure the `ssd_mobilenet.config` for the correct number of classes.**
```
& "D:/Program Files/Anaconda3/envs/tf-gpu/python.exe" "D:\Github\temp\models\research\object_detection\legacy\train.py" --logtostderr --train_dir=training/ --pipeline_config_path=ssd_mobilenet.config
```

### 4. Generate Inference Graph
```
& "D:/Program Files/Anaconda3/envs/tf-gpu/python.exe" "D:\Github\temp\models\research\object_detection\export_inference_graph.py" --input_type image_tensor --pipeline_config_path ssd_mobilenet_v2.config --trained_checkpoint_prefix training/model.ckpt --output_directory all_inference_graph
```
### 5. Run the Jupyter Notebook for test image classification
Run in $REPO/object_detection
```
jupyter notebook
```

### 6. Real-time Player detection 
Don't forget to change the inference graph folders in `player_detection.py`, as well as number of classes.
Because `player_detection.py` relies on the `utils` modules inside of `models/research/object_detection`, it needs to be copied into that folder. Also, `grabscreen.py` will need to be copied over too.

### 7. Record Supervised Learning Data
In the $REPO folder, run the following and play the game. Usually 100K+ data points is decent for training.
```
& "D:/Program Files/Anaconda3/envs/tf-gpu/python.exe" ./create_training_data.py
```

### 8. Supervised Learning Training
Run from the $REPO folder, and keep the epochs to between 5-15.
```
& "D:/Program Files/Anaconda3/envs/tf-gpu/python.exe" ./train_model.py
```

## Other

### Check if Tensorflow has detected GPU for its training sessions
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

### Anaconda Env Activation
activation involves more than simply setting PATH and we currently do not support running activate directly from PowerShell (it is on our list of things to fix). You need to use the Anaconda Prompt or, if you want to use PowerShell, run `cmd "/K" C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3` directly

### Dimensions for BBox
0,30,1280,745

### Tensorflow GPU memory issues
```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)
```

### Check memory of a particular application
```
cuda-memcheck ./yourApp
```
### Keyboard Scancodes
[Link](http://www.ee.bgu.ac.il/~microlab/MicroLab/Labs/ScanCodes.htm)

### Problem with WSAD keys not working in android emulator
Set the fixed navigation to be True.

### Accessing Private Repos in the cloud while the SSH keys are changing everytime the instance is restarted
Put the following into `~/.ssh/config`
```
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_rsa
```

Copy over the private SSH key into `~/.ssh/{PRIVATE_KEY}`

## Fine Tuning Existing models (Mobilenet as E.g.)
If you get "Expected 3 dimensions but got array with shape (BATCH_SIZE, NUM_CLASSES)"

**Before:**
The last softmax layer looks like this:
```
dense_1 (Dense)                 (None, 7, 7, 6)
```


The problem is that you start with a three dimensional layer but never reduce the dimensionality in any of the following layers.
Try adding mode.add(Flatten()) before the last Dense layer

**After:**
```
dense_1 (Dense)                 (None, 6)
```

## Visualizing CNN Feature Layers

```python
import tensorflow as tf
import tensorflow.contrib.slim as slim

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [100, 100, 3],name="x-in")

x_image = tf.reshape(x,[-1,100,100,1])
hidden_1 = slim.conv2d(x_image,5,[5,5])
pool_1 = slim.max_pool2d(hidden_1,[2,2])
hidden_2 = slim.conv2d(pool_1,5,[5,5])
pool_2 = slim.max_pool2d(hidden_2,[2,2])
hidden_3 = slim.conv2d(pool_2,20,[5,5])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[100, 100, 3],order='F')})
    plotNNFilter(units)


def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(100,100))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
    plt.show()


imageToUse = cv2.imread('object_detection/img/bounty/gameplay_3.png')[0:705, 0:1280]
img_small = cv2.resize(imageToUse, (100, 100))
getActivations(hidden_3,img_small)
```

## Try to setup an environment for reinforcement learning
Refer to [this](https://github.com/ChintanTrivedi/DeepGamingAI_FIFARL)

## Issues on the cloud Ubuntu 18.04 Server
- Tensorflow 1.13 cannot work with Cuda 10.1, use 10.0 instead
- Install python-setuptools, or else can't use `setup.py`
- If you use `easy_setup`, in other words, the `python-setuptools` above, you will need to specify the python version by using `python3` for 3+ because `python` by default will make the setup install everything in python 2.7
- You will need to do `python setup.py build` first, then `python setup.py install` after, which will install everything into the libs folder in `pyenv/version/3.6.8/libs/python3.6.8/site-packages`

## Difficulty of Transfer learning for object detection
Refer to [this](https://medium.com/nanonets/nanonets-how-to-use-deep-learning-when-you-have-limited-data-f68c0b512cab)
- Finding a large dataset to pretrain on
- Deciding which model to use for pretraining
- Difficult to debug which of the two models is not working
- Not knowing how much additional data is enough to train the model
- Difficulty in deciding where to stop using the pretrained model
- Deciding the numer of layers and number of parameters in the model used on top of the pretrained model
- Hosting and serving the combined models
- Updating the pretrained model when more data or better techniques becomes available

# Performance Tracking

## Run 1 (1e-4_354EP_128Batch_TrainPer256Steps)
```
LEARNING_RATE = 1e-4
EP = 354
BATCH_SIZE=128
TRAIN_PER_STEPS = 256
```
Summary:
Perhaps the learning rate was too small, after 354 episodes, the mean cost of both movement and attack networks was still around 9500.
The mean reward per episode did not show an increase trend over time, it was still revolving around 0.2 with large variances.

## Run 2 (3)
```
LEARNING_RATE = 3e-3
EP = 500
BATCH_SIZE=128
TRAIN_PER_STEPS = 256
```
Summary:
After increasing the learning rate from 1e-4 to 3e-3, after 500 episodes, the mean cost of both movement and attack networks is around 785.
The mean reward per episode showed an increase trend over time and peaked around EP 212, and decreased and stablized around 0.18. The agent's behavior in game is basically standing in the corner attacking the air once every 5 seconds. It must be because the agent realized the high cost of attacking, and experienced an increase in rewards after decreasing the attack frequency.


## Run 3 (4)
```
LEARNING_RATE = 3e-3
EP = 727
BATCH_SIZE=128
TRAIN_PER_STEPS = 256
```
The cost decreased to 555 after 727 episodes, however the average reward is still oscilating around 0.05.

## Run 4 (5)
```
LEARNING_RATE = 3e-3
EP = 1000
BATCH_SIZE=128
TRAIN_PER_STEPS = 256
```
The cost decreased to 22 after 1000 episodes, however the average reward is still oscilating around 0.35. I've noticed the agent not moving at all for prolonged periods of time, and suddenly moving forward even keep pressing forward after reaching the opposing spawn point's wall.