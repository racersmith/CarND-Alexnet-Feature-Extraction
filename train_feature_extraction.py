import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle

# TODO: Load traffic signs data.
with open('train.p', 'rb') as f:
	data = pickle.load(f)

X_train = data['features']
y_train = data['labels']

# TODO: Split data into training and validation sets.
X_train, y_train = shuffle(X_train, y_train)
validation_ratio = 0.2
X_train, X_validation, y_train, y_validation = train_test_split(
	X_train,
	y_train,
	test_size=validation_ratio,
	random_state=121)

# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(features, [227, 227])
labels = tf.placeholder(tf.int64, (None))
nb_classes = 43
one_hot_y = tf.one_hot(labels, nb_classes)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.

shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

fc8W = tf.Variable(tf.truncated_normal(shape, stddev=0.01))
fc8b = tf.Variable(tf.zeros(nb_classes))

logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# HYPERPERAMETERS
RATE = 0.001
NO_IMPROVEMENT_STOP = 3      # Stop after no improvement over N Epochs
BATCH_SIZE = 128

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = RATE)
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])
prediction = tf.arg_max(logits, 1)
accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset: offset + BATCH_SIZE], y_data[offset: offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={features: batch_x, labels: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# TODO: Train and evaluate the feature extraction model.
saver = tf.train.Saver()
print("Starting Tensorflow Session")
with tf.Session() as sess:
    print("Session started")
    sess.run(tf.global_variables_initializer())
    print("Variables initialized")
    num_examples = len(X_train)
    accuracy_history = []
    
    print("Training...")
    print()

    keep_learning = True
    while keep_learning:
        X_train, y_train = shuffle(X_train, y_train)
        print("shuffled")
        for offset in range(0, X_train.shape[0], BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={features: batch_x, labels: batch_y,})
        print("start validation")
        validation_accuracy = evaluate(X_validation, y_validation)
        accuracy_history.append(validation_accuracy)
        
        # check for accuracy improvement
        if len(accuracy_history) > NO_IMPROVEMENT_STOP:
            keep_learning = max(accuracy_history[-NO_IMPROVEMENT_STOP:]) > max(accuracy_history[:-NO_IMPROVEMENT_STOP])
        
        print("EPOCH {} ...".format(len(accuracy_history)))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    
    saver.save(sess, "tmp/model")
    
    print("Model saved")
