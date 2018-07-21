# -*- coding: utf-8 -*-

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

#DB 및 파일 저장 구현

#############################################
#시작 DB에서 WORD_DIRCTIONAY 읽어 오기


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

#트레이닝 준비끝 
import tensorflow as tf

#입력데이터 로우버퍼
x = tf.placeholder(tf.float32, [None, len(train_dataset[0])])

#연산 부분 버퍼 (입력데이터->패턴인식->결과:확률)
W = tf.Variable(tf.zeros([len(train_dataset[0]), len(tarin_labelset[0])]))
b = tf.Variable(tf.zeros([len(tarin_labelset[0])]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, len(tarin_labelset[0])]) 

#딥러닝 세션 오픈
sess = tf.Session()


#덤프 된 파일 불러오기
saver = tf.train.Saver()
saver.restore(sess, "C:/ailing/0/model.ckpt")

chat_text = '학생부 성적 반영방법은 어떻게 되나요?'
chat_word = nlp.nouns(chat_text)
chat_data = np.zeros(len(word_dictionary))
chat_dataset = []

for i in range(0, len(word_dictionary)):
    for word in chat_word:
        if(word_dictionary[i]==word):
            chat_data[i] = 1
    
chat_dataset.append(list(chat_data))

print(chat_dataset)

result = sess.run(y, feed_dict={x: chat_dataset})
print(list(result))
print(list(result[0]))