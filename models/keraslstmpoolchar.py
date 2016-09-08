'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import numpy as np, csv, keras
from IPython import embed
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from nltk.tokenize import word_tokenize as tokenize

class _GlobalPooling1D(Layer):

    def __init__(self, **kwargs):
        super(_GlobalPooling1D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]
        #self.supports_masking = True

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        raise NotImplementedError


class GlobalMaxPooling1D(_GlobalPooling1D):
    '''Global average pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    '''

    def compute_mask(self, x, mask):
        return None

    def call(self, x, mask=None):
        ret = K.max(x, axis=1)
        return ret
        #sum = K.sum(x, axis=1)      # (samples, features)
        #tot = K.sum(mask, axis=1) if mask is not None else x.shape[1]  # (samples, )
        #return sum / tot

print(keras.__version__)

max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 20
mode = "char"

def readdata(trainp, testp, mode=None, masksym=-1, maxlen=100):
    assert(mode is not None)
    if mode is "char":
        return readdata_char(trainp, testp, maxlen=maxlen, masksym=masksym)
    elif mode is "word":
        return readdata_word(trainp, testp, maxlen=maxlen, masksym=masksym)


def readdata_word(trainp, testp, maxlen=100, masksym=-1):
    def readdataset(p, wdic, maxlen=100):
        dataret = []
        goldret = []
        toolong = 0
        realmaxlen = 0
        wdic[None] = masksym
        with open(p) as f:
            data = csv.reader(f, delimiter=",")
            for row in data:
                rowelems = tokenize(row[2])
                realmaxlen = max(realmaxlen, len(rowelems))
                if len(rowelems) > maxlen:
                    toolong += 1
                for rowelem in set(rowelems):
                    if rowelem not in wdic:
                        wdic[rowelem] = len(wdic)
                dataret.append([wdic[x] for x in rowelems])
                goldret.append(row[0])
        print("{} comments were too long".format(toolong))
        maxlen = min(maxlen, realmaxlen)
        datamat = np.ones((len(dataret) - 1, maxlen)).astype("int32") * masksym
        for i in range(1, len(dataret)):
            datamat[i - 1, :min(len(dataret[i]), maxlen)] = dataret[i][:min(len(dataret[i]), maxlen)]
        return datamat, np.asarray(goldret[1:], dtype="int32"), wdic

    traindata, traingold, wdic = readdataset(trainp, {}, maxlen=maxlen)
    testdata, testgold, wdic = readdataset(testp, wdic=wdic, maxlen=maxlen)
    return (traindata, traingold), (testdata, testgold), wdic


def readdata_char(trainp, testp, maxlen=1000, masksym=-1):
    def readdataset(p):
        dataret = []
        goldret = []
        toolong = 0
        with open(p) as f:
            data = csv.reader(f, delimiter=",")
            for row in data:
                if len(row[2]) > maxlen:
                    toolong += 1
                dataret.append([ord(x) for x in row[2]])
                goldret.append(row[0])
        print("{} comments were too long".format(toolong))
        datamat = np.ones((len(dataret)-1, maxlen)).astype("int32") * masksym
        for i in range(1, len(dataret)):
            datamat[i-1, :min(len(dataret[i]), maxlen)] = dataret[i][:min(len(dataret[i]), maxlen)]
        return datamat, np.asarray(goldret[1:], dtype="int32")
    traindata, traingold = readdataset(trainp)
    testdata, testgold = readdataset(testp)
    allchars = set(list(np.unique(traindata))).union(set(list(np.unique(testdata))))
    chardic = dict(zip(list(allchars), range(len(allchars))))
    chardic[masksym] = masksym
    traindata = np.vectorize(lambda x: chardic[x])(traindata)
    testdata = np.vectorize(lambda x: chardic[x])(testdata)
    chardic = {chr(k): v for k, v in chardic.items() if k != masksym}
    return (traindata, traingold), (testdata, testgold), chardic

# load data
(traindata, traingold), (testdata, testgold), dic = readdata("../data/kaggle/train.csv", "../data/kaggle/test_with_solutions.csv",
                                                             mode=mode, masksym=0, maxlen=maxlen if mode == "word" else maxlen*8)

# subsample for balancing
posindexes = np.argwhere(traingold)
negindexes = np.argwhere(1-traingold)
allindexes = sorted(list(posindexes[:, 0]) + list(negindexes[:posindexes.shape[0], 0]))
subtraindata = traindata[allindexes, :]
subtraingold = traingold[allindexes]

#embed()
print('Build model...')
model = Sequential()
model.add(Embedding(len(dic)+1, 300, dropout=0.2, mask_zero=True))
model.add(LSTM(300, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
model.add(LSTM(300, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
#model.add(LSTM(300, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
model.add(GlobalMaxPooling1D())
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(subtraindata, subtraingold, batch_size=batch_size, nb_epoch=30,
          validation_data=(testdata, testgold))
score, acc = model.evaluate(testdata, testgold,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
