import tensorflow as tf
import numpy as np
import cPickle as p 
from PIL import Image
import random
def buildgraph(): 
	return 0

minibatch = 128
size = (200,200)

class Reader:
  def __init__(self, path):
    lines = []
    with open(path, "rb") as file:
      lines = file.readlines()
    
    random.shuffle(lines)

    self.all_x,self.all_y = [(l.split()[0]) for l in lines],[(l.split()[1]) for l in lines]
    self.x = self.all_x[:7000]
    self.y = self.all_y[:7000]

    self.dev_x = self.all_x[7000:8000]
    self.dev_y = self.all_y[7000:8000]
    self.dev_i = 0

  def getBatch(self):
    def oneHot(s): 
      if (s == "circle"): return np.array([1, 0, 0])
      if (s == "rectangle"): return np.array([0, 1, 0])
      if (s == "roundRectangle"): return np.array([0, 0, 1])

    def nameToImage(name):
      im = Image.open(name)
      im.load()
      return np.asarray(im, dtype="float32")
      
    indices = np.random.choice(range(minibatch), minibatch)
    images = np.array([nameToImage("images/" + self.x[i]) for i in indices])
    labels = np.array([oneHot(self.y[i]) for i in indices])
    return (images,labels)
 

class CNN: 
  def __init__(self):

    filters = [1,2,3,5,7]
    numFilters = 32
    numIn = 4
    pools = []
    

    self.x = tf.placeholder(tf.float32, shape=[minibatch,200,200,numIn])
    self.y = tf.placeholder(tf.int32, shape=[minibatch,3])
    self.keep_prob = tf.placeholder(tf.float32)    

    for filter in filters:
      with tf.name_scope("conv" + str(filter)):
        filterShape = [filter, filter, numIn, numFilters]
        W = tf.Variable(tf.truncated_normal(filterShape, stddev = .1), name="W")
        b = tf.Variable(tf.constant(.1, shape=[numFilters], name="b"))
        conv = tf.nn.conv2d(self.x, W, strides=[1,1,1,1], padding="VALID", name="conv")
        h = tf.nn.relu(tf.nn.bias_add(conv, b, name="relu"))
        pool = tf.nn.max_pool(h, ksize=[1, 200-filter+1, 200-filter+1, 1], strides=[1,1,1,1], padding="VALID", name="pool")
        pools.append(pool)
    allpools = tf.concat(3, pools)
    hsize = numFilters * len(filters)
    final_pool = tf.reshape(allpools, [-1, hsize]  )
    h_drop     = tf.nn.dropout(final_pool, self.keep_prob)

    softmax_w = tf.Variable(tf.truncated_normal( [hsize  , 3], stddev=.1 ), name="softmax_w")
    softmax_b = tf.Variable(tf.constant(.1, shape=[3]), name="softmax_b")
    logits = tf.matmul( h_drop, softmax_w ) + softmax_b

    self.loss =  tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, self.y, name="softmax_op") )
    
    self.opt  = tf.train.AdamOptimizer().minimize(self.loss)
 
def main(_):
    r = Reader("images/labels.txt")
    cnn = CNN()
    maxSteps = 10000
    sv = tf.train.Supervisor(logdir="logs", summary_op=None)
    with sv.managed_session() as sess:
	   # sess.run ( tf.global_variables_initializer())

            for step in range(maxSteps):
	        batch_x, batch_y = r.getBatch()
	        feed_dict = { cnn.x: batch_x, cnn.y: batch_y, cnn.keep_prob:.75}
	        fetch = { "loss":cnn.loss, "opt":cnn.opt}
	        res = sess.run(fetch, feed_dict=feed_dict)
	        if step % 54 == 0:
                    epoch = step/54
	  	    print "epoch:", epoch, "loss:", res['loss']
                    sv.saver.save(sess, 'inferProgram', global_step=epoch)

if (__name__ == "__main__"): 
  tf.app.run()
