## Windows

### 1. Convert XML labels to CSV

Run in $REPO/object_detection
```
& "D:/Program Files/Anaconda3/envs/py-35/python.exe" d:/Github/brawlstars-ai/object_detection/xml_to_csv.py
```

### 2. Generate TF_Record

Run in $REPO/object_detection
```
& "D:/Program Files/Anaconda3/envs/py-35/python.exe" .\generate_tfrecord.py --csv_input=data/train.csv --output_path=data/train.record --image_dir=img/train
```

```
& "D:/Program Files/Anaconda3/envs/py-35/python.exe" .\generate_tfrecord.py --csv_input=data/test.csv --output_path=data/test.record --image_dir=img/test
```

### 3. Train Object Detection Model
Run in $REPO/object_detection
```
& "D:/Program Files/Anaconda3/envs/py-35/python.exe" "D:\Github\temp\models\research\object_detection\legacy\train.py" --logtostderr --train_dir=training/ --pipeline_config_path=ssd_mobilenet.config
```

### 4. Generate Inference Graph
```
& "D:/Program Files/Anaconda3/envs/tf-gpu/python.exe" "D:\Github\temp\models\research\object_detection\export_inference_graph.py" --input_type image_tensor --pipeline_config_path ssd_mobilenet.config --trained_checkpoint_prefix training/model.ckpt-20000 --output_directory name_inference_graph
```
### 5. Run the Jupyter Notebook for test image classification
Run in $REPO/object_detection
```
jupyter notebook
```

### 6. Real-time Player detection 
Because `player_detection.py` relies on the `utils` modules inside of `models/research/object_detection`, it needs to be copied into that folder. Also, `grabscreen.py` will need to be copied over too.

### 7. Record Supervised Learning Data
In the $REPO folder, run the following and play the game. Usually 100K+ data points is decent for training.
```
"D:/Program Files/Anaconda3/envs/py-35/python.exe" ./create_training_data.py
```


### 8. Supervised Learning Training
Run from the $REPO folder, and keep the epochs to between 5-15.
```
"D:/Program Files/Anaconda3/envs/py-35/python.exe" ./train_model.py
```

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

### Accessing Private Repos in the cloud while the SSH keys are changing everytime the instance is restarted
Put the following into `~/.ssh/config`
```
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_rsa
```

Copy over the private SSH key into `~/.ssh/{PRIVATE_KEY}`


## Issues on the cloud Ubuntu 18.04 Server
- Tensorflow 1.13 cannot work with Cuda 10.1, use 10.0 instead
- Install python-setuptools, or else can't use `setup.py`
- If you use `easy_setup`, in other words, the `python-setuptools` above, you will need to specify the python version by using `python3` for 3+ because `python` by default will make the setup install everything in python 2.7
- You will need to do `python setup.py build` first, then `python setup.py install` after, which will install everything into the libs folder in `pyenv/version/3.6.8/libs/python3.6.8/site-packages`