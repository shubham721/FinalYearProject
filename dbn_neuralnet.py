from dbn_outside.dbn.tensorflow import SupervisedDBNClassification
from sentiment_analysis import create_feature_set_and_labels
import numpy as np
from sklearn.metrics.classification import accuracy_score
train_x,train_y,test_x,test_y = create_feature_set_and_labels('pos_hindi.txt','neg_hindi.txt')
train_x = np.array(train_x,dtype=np.float32)
train_y = np.array(train_y,dtype=np.float32)
test_x = np.array(test_x,dtype=np.float32)
test_y = np.array(test_y,dtype=np.float32)

classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)

classifier.fit(train_x,train_y)
classifier.save('dbn.pkl')
y_pred = classifier.predict(test_x)
accuracy = accuracy_score(test_y,y_pred)
print(accuracy)

                                        