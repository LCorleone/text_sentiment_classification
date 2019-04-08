'''
word embedding测试
在GTX960上，18s一轮
经过30轮迭代，训练集准确率为98.41%，测试集准确率为89.03%
Dropout不能用太多，否则信息损失太严重
'''
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
import numpy as np
import pandas as pd
import jieba
import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)
KTF.set_session(session)


pos = pd.read_excel('./data/pos.xls', header=None)
pos['label'] = 1
neg = pd.read_excel('./data/neg.xls', header=None)
neg['label'] = 0
all_ = pos.append(neg, ignore_index=True)
all_['words'] = all_[0].apply(lambda s: list(jieba.cut(s)))  # 调用结巴分词

maxlen = 100  # 截断词数
min_count = 5  # 出现次数少于该值的词扔掉。这是最简单的降维方法

content = []
for i in all_['words']:
    content.extend(i)

abc = pd.Series(content).value_counts()
abc = abc[abc >= min_count]
abc[:] = list(range(1, len(abc) + 1))
abc[''] = 0  # 添加空字符串用来补全
word_set = set(abc.index)


def doc2num(s, maxlen):
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + [''] * max(0, maxlen - len(s))
    return list(abc[s])


all_['doc2num'] = all_['words'].apply(lambda s: doc2num(s, maxlen))

# 手动打乱数据
idx = list(range(len(all_)))
np.random.shuffle(idx)
all_ = all_.loc[idx]

# 按keras的输入要求来生成数据
x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
y = y.reshape((-1, 1))  # 调整标签形状

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM

# 建立模型
model = Sequential()
model.add(Embedding(len(abc), 256, input_length=maxlen))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
batch_size = 128
train_num = 15000


if not os.path.exists('./models'):
    os.makedirs('./models')
model_name = 'word_embedding_model_ex-{epoch:03d}_val_acc-{val_acc:03f}.h5'
model_path = os.path.join('models', model_name)
checkpoint = ModelCheckpoint(filepath=model_path,
                             monitor='acc',
                             verbose=1,
                             save_weights_only=True,
                             period=1)

model.fit(x[:train_num], y[:train_num], batch_size=batch_size, epochs=30, validation_data=(x[train_num:], y[train_num:]),
          callbacks=[checkpoint])


def predict_one(s):  # 单个句子的预测函数
    s = np.array(doc2num(list(jieba.cut(s)), maxlen))
    s = s.reshape((1, s.shape[0]))
    return model.predict_classes(s, verbose=0)[0][0]
