"""
lab1/addition by @PV Pho Vu 
date: may 17, 2022
"""

import os # set up environment
import tensorflow as tf # needs library

tf.compat.v1.disable_eager_execution()

"""
we can turn off TensorFlow (TF) warning messages in program output
the code is created to prevent TensorFlow from outputting as many log messages as it normally does
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# define computational graph
# tf v1 -> tf v 2.5 | activate v1 in tf.placeholder
X = tf.compat.v1.placeholder(tf.float32, name="X")
Y = tf.compat.v1.placeholder(tf.float32, name="Y")

# variable for addition
addition = tf.add(X, Y, name="addition")

# operate the session
# tf.Session() no longer works | incorporate compat.v1 into tf.Session()
with tf.compat.v1.Session() as session:

    result = session.run(addition, feed_dict={X: [1], Y: [4]})

    print(result) # run the program
