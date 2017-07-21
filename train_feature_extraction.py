import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.
data_f = './train.p'

with open(data_f, mode='rb') as f:
    data = pickle.load(f)

X_data = data['features']
y_data = data['labels']

# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=21)
print("train shape:", X_train.shape)
print("train label shape:", y_train.shape)
print("test shape:", X_test.shape)
print("train 1st sample shape:", X_train[0].shape)
print("image channel:", X_train[0][0][0].shape)

X_train_rest, X_train_sub, y_train_rest, y_train_sub = train_test_split(X_train, y_train, test_size=0.02, random_state=42)
X_test_rest, X_test_sub, y_test_rest, y_test_sub = train_test_split(X_test, y_test, test_size=0.02, random_state=42)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
nb_classes = 43
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc_W = tf.Variable(tf.truncated_normal(shape, mean=0, stddev=0.1))
fc_b = tf.Variable(tf.zeros(nb_classes))
logits = tf.matmul(fc7, fc_W) + fc_b

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
EPOCHS = 10
BATCH_SIZE = 128
learning_rate = 0.001

y = tf.placeholder(tf.int32, (None), name="placeholder_y")
one_hot_y = tf.one_hot(y, nb_classes)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_y, logits = logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
# Evaluate the model
def evaluate(X_data, y_data):    
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# TODO: Train and evaluate the feature extraction model.
from sklearn.utils import shuffle

# preprocess the data
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training the model...")
    print()
    for i in range(EPOCHS):
        # Train the model
        X_train_sub, y_train_sub = shuffle(X_train_sub, y_train_sub)
        for start in range(0, num_examples, BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_x, batch_y = X_train_sub[start:end], y_train_sub[start:end]
            sess.run(train_operation, feed_dict = {x: batch_x, y: batch_y})
            
        # Evaluate the model on the validation data set
        train_accuracy = evaluate(X_train_sub, y_train_sub)
        print("EPOCH {} ...".format(i+1))
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        
    # Evaluate the model on the test data set
    test_accuracy = evaluate(X_test_sub, y_test_sub)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    print()
