import os # set up environment
import tensorflow as tf # needs library

tf.compat.v1.disable_eager_execution() #v1 -> v-2.5

# we can turn off TensorFlow (TF) warning messages in program output
# prevent TensorFlow from outputting as many log messages as it normally does
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# define computational graph
X = tf.compat.v1.placeholder(tf.float32, name="X")
Y = tf.compat.v1.placeholder(tf.float32, name="Y")

addition = tf.add(X, Y, name="addition")

# operate the session
with tf.compat.v1.Session() as session:

    result = session.run(addition, feed_dict={X: [1, 2, 10], Y: [4, 2, 10]}) # matrice

    print(result) # run the program
