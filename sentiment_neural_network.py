import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot= True)
from sentiment_analysis import create_feature_set_and_labels

train_x,train_y,test_x,test_y = create_feature_set_and_labels('pos_hindi.txt','neg_hindi.txt')

hidden_layer_1_nodes = 500 #nodes in hidden layer
hidden_layer_2_nodes = 500
hidden_layer_3_nodes = 500

output_classes = 2
batch_size = 100

x = tf.placeholder('float',[None,len(train_x[0])])  #28*28 pixels (if you doesn't specify 2nd argument tensorflow handle it for you)
y = tf.placeholder('float')


def neural_network_model(data):
    #(input_data*weights)+bias

    layer_1 = {'weights':tf.Variable(tf.random_normal([len(train_x[0]),hidden_layer_1_nodes])),
                'biases':tf.Variable(tf.random_normal([hidden_layer_1_nodes]))}

    layer_2 = {'weights':tf.Variable(tf.random_normal([hidden_layer_1_nodes,hidden_layer_2_nodes])),
                'biases':tf.Variable(tf.random_normal([hidden_layer_2_nodes]))}

    layer_3 = {'weights':tf.Variable(tf.random_normal([hidden_layer_2_nodes,hidden_layer_3_nodes])),
                'biases':tf.Variable(tf.random_normal([hidden_layer_3_nodes]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([hidden_layer_3_nodes,output_classes])),
                'biases':tf.Variable(tf.random_normal([output_classes]))}

    # model = (input_data*weights)+bias

    l1 = tf.add(tf.matmul(data,layer_1['weights']), layer_1['biases'])
    l1 = tf.nn.relu(l1) #kind of a activation function

    l2 = tf.add(tf.matmul(l1,layer_2['weights']),layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,layer_3['weights']),layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,output_layer['weights']),output_layer['biases'])

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    #this is cost function which estimates how fare away we are from actual output
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # now we want to minimize the cost using optimization function
    # AdamOptimizer has one optional attribute learning_rate which is default to 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # epochs = feed_forward + back propagation
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while(i<len(train_x)):
                start = i
                end = i+batch_size
                epoch_x = train_x[start:end] 
                epoch_y = train_y[start:end]
                #epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer,cost], feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch ',epoch,'Completd out of ',hm_epochs,' loss: ',epoch_loss)
        # the above is whole training part

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        print(correct)
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('ha' ,accuracy)
        print('Accuracy: ',accuracy.eval({x:test_x,y:test_y})*100,"%")


train_neural_network(x)
