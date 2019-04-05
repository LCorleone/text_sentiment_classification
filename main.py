
'''
one hot测试
在GTX960上，约100s一轮
经过90轮迭代，训练集准确率为96.60%，测试集准确率为89.21%
Dropout不能用太多，否则信息损失太严重
'''
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
import numpy as np
import pandas as pd
import pdb
import os
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


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

maxlen = 200  # 截断字数
min_count = 20  # 出现次数少于该值的字扔掉。这是最简单的降维方法

content = ''.join(all_[0])
abc = pd.Series(list(content)).value_counts()  # 统计每个字出现的次数
abc = abc[abc >= min_count]  # 出现小于20个字的剔除掉
abc[:] = list(range(len(abc)))  # abc是字与编号的对应表
word_set = set(abc.index)  # 字表


def doc2num(s, maxlen):
    s = [i for i in s if i in word_set]
    s = s[:maxlen]
    return list(abc[s])


all_['doc2num'] = all_[0].apply(lambda s: doc2num(s, maxlen))

# 手动打乱数据
# 当然也可以把这部分加入到生成器中
idx = list(range(len(all_)))
np.random.shuffle(idx)
all_ = all_.loc[idx]

# 按keras的输入要求来生成数据
x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
y = y.reshape((-1, 1))  # 调整标签形状

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import sys
sys.setrecursionlimit(10000)  # 增大堆栈最大深度(递归深度)，据说默认为1000，报错

# 建立模型
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(abc))))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 单个one hot矩阵的大小是maxlen*len(abc)的，非常消耗内存
# 为了方便低内存的PC进行测试，这里使用了生成器的方式来生成one hot矩阵
# 仅在调用时才生成one hot矩阵
# 可以通过减少batch_size来降低内存使用，但会相应地增加一定的训练时间
batch_size = 512
train_num = 15000

# 不足则补全0行


def gen_matrix(z):
    return np.vstack((np_utils.to_categorical(z, len(abc)), np.zeros((maxlen - len(z), len(abc)))))


def data_generator(data, labels, batch_size):
    batches = [list(range(batch_size * i, min(len(data), batch_size * (i + 1))))
               for i in range(int(len(data) / batch_size + 1))]
    while True:
        for i in batches:
            xx = np.zeros((maxlen, len(abc)))
            xx, yy = np.array(list(map(gen_matrix, data[i]))), labels[i]
            yield (xx, yy)


if not os.path.exists('./models'):
    os.makedirs('./models')
model_name = 'model_ex-{epoch:03d}_val_acc-{val_acc:03f}.h5'
model_path = os.path.join('models', model_name)
checkpoint = ModelCheckpoint(filepath=model_path,
                             monitor='acc',
                             verbose=1,
                             save_weights_only=True,
                             period=1)


model.fit_generator(data_generator(
    x[:train_num], y[:train_num], batch_size), steps_per_epoch=train_num // batch_size, epochs=30, callbacks=[checkpoint],
    validation_data=data_generator(x[train_num:], y[train_num:], batch_size), validation_steps=len(x[train_num:]) // batch_size)

# model.evaluate_generator(data_generator(
#     x[train_num:], y[train_num:], batch_size), val_samples=len(x[train_num:]))


def predict_one(s):  # 单个句子的预测函数
    s = gen_matrix(doc2num(s, maxlen))
    s = s.reshape((1, s.shape[0], s.shape[1]))
    return model.predict_classes(s, verbose=0)[0][0]
