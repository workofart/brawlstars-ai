import time, numpy as np
import cv2, os
import tensorflow as tf
from utilities.directkeys import PressKey, ReleaseKey, Q, W, E, S, A, D
from utilities.window import WindowMgr

def superattack():
    PressKey(Q)
    time.sleep(0.05)
    ReleaseKey(Q)

def attack():
    PressKey(E)
    time.sleep(0.05)
    ReleaseKey(E)

def releaseAllKeys():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def front():
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    PressKey(W)

def left():
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    PressKey(A)

def right():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    PressKey(D)
    
def back():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    PressKey(S)

def countdown(t):
    for i in list(range(t))[::-1]:
        print(i+1)
        time.sleep(1)

def mouse(img):
    ix,iy = -1,-1
    # mouse callback function
    def draw_circle(event,x,y,flags,param):
        global ix,iy
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img,(x,y),100,(255,0,0),-1)
            ix,iy = x,y
            print(ix,iy)

    cv2.setMouseCallback('window',draw_circle)

def take_action(movement_index, action_index):
    if movement_index == 0:
        # print('left')
        left()
    elif movement_index == 1:
        # print('front')
        front()
    if movement_index == 2:
        # print('right')
        right()
    elif movement_index == 3:
        # print('back')
        back()
    elif movement_index == 4:
        releaseAllKeys()
    time.sleep(0.2)
    if action_index == 0:
        # print('attack')
        attack()
    elif action_index == 1:
        # print('superattack')
        superattack()
    movement_map = {
        0: A,
        1: W,
        2: D,
        3: S,
        4: ''
    }

    action_map = {
        0: E,
        1: Q,
        2: ''
    }

    
def variable_summaries(var):
    out = []
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    # Taken from https://www.tensorflow.org/guide/summaries_and_tensorboard
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        out.append(tf.summary.scalar(var.op.name + '_mean', mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        out.append(tf.summary.scalar(var.op.name + '_stddev', stddev))
        out.append(tf.summary.scalar(var.op.name + '_max', tf.reduce_max(var)))
        out.append(tf.summary.scalar(var.op.name + '_min', tf.reduce_min(var)))
        out.append(tf.summary.histogram(var.op.name + '_histogram', var))
        return out

def log_histogram(writer, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)
        
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        writer.add_summary(summary, step)
        writer.flush()

def log_scalars(writer, tag, values, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=values)])
    writer.add_summary(summary, step)
    writer.flush()


def get_latest_run_count():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    dirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    if len(dirs) == 0:
        return 0
    else:
        return int(max(dirs)) + 1

