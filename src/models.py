import tensorflow as tf 
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPool1D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras import Model 

class vggrepeat(Model):
    def __init__(self):
        super(vggrepeat, self).__init__()
        self.conv = Conv1D(filters=64, kernel_size=3)
        self.batch = BatchNormalization()
        self.pool = MaxPool1D(pool_size=2)

    def call(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.pool(x)

        return x

class vgg(Model):
    def __init__(self):
        super(vgg, self).__init__()
        numlayers = 5
        self.repeatlayers = [vggrepeat() for _ in range(numlayers)]
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='selu')
        self.dr = Dropout(0.5)
        self.d2 = Dense(128, activation='selu')
        self.d3 = Dense(5, activation='softmax')

    def call(self, x):
        for layer in self.repeatlayers:
            x = layer(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.dr(x)
        x = self.d2(x)

        return self.d3(x)