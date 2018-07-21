import numpy as np
import os
import pandas as pd
from collections import Counter
from konlpy.tag import Twitter

dir = 'C:/chatdata'

loc = os.listdir(dir);

content = []
nouns = []

# Listing Texts
for maindir in loc:
    subdir = os.listdir(dir + '/' + maindir)
    file_list = []
    
    for file in subdir:
        file_list.append(open(dir + '/' + maindir + '/' + file, "r").read())
    
    content.append(file_list)

nlp = Twitter();

# Seperating Words
for i in content:
    list_wrap = []
    for j in i:
        list_wrap.append(nlp.nouns(j))
        
    nouns.append(list_wrap)
    
words = ''
    
for c in content[0]:
    words = words + ' ' + c

nouns2 = nlp.nouns(words)
count2 = Counter(nouns2)

tags = count2.most_common()

taglist = np.array(tags)
word_dictionary = taglist[0:, 0]

#DB �� ���� ���� ����

#############################################
#���� DB���� WORD_DIRCTIONAY �о� ����


label = []
training = []

import numpy as np
import os
import pandas as pd

dir = 'C:/chatdata'
loc = os.listdir(dir);
                
print(len(loc))                
# Listing Texts
for maindir in loc:
    subdir = os.listdir(dir + '/' + maindir)
    file_list = []
    
    for file in subdir:
        file_list = open(dir + '/' + maindir + '/' + file, "r").read()
        label.append(maindir)
        training.append(file_list)

print(len(label))
print(len(training))

print(label[0])
print(training[0])

train_dataset = []

for i in training:
    train_word = nlp.nouns(i)
    train_data = np.zeros(len(word_dictionary))
    for i2 in range(0,len(word_dictionary)):
        for word in train_word:
            if(word_dictionary[i2]==word):
                train_data[i2] = 1

    train_dataset.append(list(train_data))
    

tarin_labelset = []



for i in label:
    train_label = np.zeros(len(loc))
    train_label[(int(i)-1)] = 1
    tarin_labelset.append(list(train_label))

print(len(label))
print(len(training))
print(len(train_dataset))

print(len(tarin_labelset))
print(len(tarin_labelset[0]))

print(len(train_dataset[0]))
print(len(word_dictionary))

print(train_dataset[0])


x = train_dataset
y = label

#Ʈ���̴� �غ� 
import tensorflow as tf

#�Էµ����� �ο����
x = tf.placeholder(tf.float32, [None, len(train_dataset[0])])

#���� �κ� ���� (�Էµ�����->�����ν�->���:Ȯ��)
W = tf.Variable(tf.zeros([len(train_dataset[0]), len(tarin_labelset[0])]))
b = tf.Variable(tf.zeros([len(tarin_labelset[0])]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, len(tarin_labelset[0])]) 

#�н� ��Ƽ����¡
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


#������ ���� ����
sess = tf.Session()
#���� �ʱ�ȭ
sess.run(tf.global_variables_initializer())

#Ʈ���̴� ����
for i in range(1000):
  sess.run(train_step, feed_dict={x: train_dataset, y_: tarin_labelset})
  print(i)


#Ʈ���̴� �򰡸�
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#{x: train_dataset, y_: label} => �׽�Ʈ�� ������ ������ �غ� �ʿ�.
print(sess.run(accuracy, feed_dict={x: train_dataset, y_: tarin_labelset}))
#��Ȯ��

#dump ����
saver = tf.train.Saver()

save_path = saver.save(sess, "C:/ailing/0/model.ckpt")

