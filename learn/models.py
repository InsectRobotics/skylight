from keras.models import Model
from keras.layers import Dense, Conv2D, Input, Dropout, Flatten


class CompassModel(Model):

    def __init__(self, name="SkyCompass"):
        inp = Input((1, 104, 473), name="SkyFeatures")
        out = Conv2D(10, (1, 473), data_format='channels_first', activation='relu', name="PixelConv10")(inp)
        out = Flatten(name="PixelHidden0")(out)
        out = Dense(500, activation='relu', name="PixelHidden1")(out)
        out = Dropout(0.5, name="Dropout1-0.5")(out)
        out = Dense(100, activation='relu', name="PixelHidden2")(out)
        out = Dense(100, activation='relu', name="PixelHidden3")(out)
        out = Dense(100, activation='relu', name="PixelHidden4")(out)
        out = Dropout(0.5, name="Dropout2-0.5")(out)
        out = Dense(8, activation='tanh', name="Compass")(out)

        super(CompassModel, self).__init__(inp, out, name=name)
