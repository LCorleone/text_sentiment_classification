import numpy as np
import pandas as pd
import pdb
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import codecs
# 必须事先知道文件的编码格式，这里文件编码是使用的utf-8
f = codecs.open('./data/pos.txt', 'r+', encoding='utf-8')
content = f.read()  # 如果open时使用的encoding和文件本身的encoding不一致的话，那么这里将将会产生错误
print(content[0:300])
