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
import numpy as np
import sys
sys.setrecursionlimit(10000)  # 增大堆栈最大深度(递归深度)，据说默认为1000，报错


def gen_matrix(z):
    return np.vstack((np_utils.to_categorical(z, len(abc)), np.zeros((maxlen - len(z), len(abc)))))


def predict_one(s, model):  # 单个句子的预测函数
    s = gen_matrix(doc2num(s, maxlen))
    s = s.reshape((1, s.shape[0], s.shape[1]))
    return model.predict_classes(s, verbose=0)[0][0]


maxlen = 200  # 截断字数
min_count = 20  # 出现次数少于该值的字扔掉。这是最简单的降维方法
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

model_path = './models/model_ex-021_val_acc-0.774897.h5'
model.load_weights(model_path)

print(predict_one('这个东西，质量太好了，用着很舒服', model))
