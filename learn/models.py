from keras.models import Model
from keras.layers import Dense, Conv2D, Input, Dropout, Flatten


class BlackbodyModel(Model):

    def __init__(self, name="BlackbodyCompass"):
        inp = Input((1, 104, 473), name="SkyFeatures")
        out = Conv2D(10, (1, 473), data_format='channels_first', activation='relu', name="PixelConv10")(inp)
        self.filter = Model(inp, out, name="%s-filter" % name)

        out = Flatten(name="PixelHidden0")(out)
        out = Dense(500, activation='relu', name="PixelHidden1")(out)
        out = Dropout(0.5, name="Dropout1-0.5")(out)
        out = Dense(100, activation='relu', name="PixelHidden2")(out)
        out = Dense(100, activation='relu', name="PixelHidden3")(out)
        out = Dense(100, activation='relu', name="PixelHidden4")(out)
        out = Dropout(0.5, name="Dropout2-0.5")(out)
        out = Dense(8, activation='tanh', name="Compass")(out)

        super(BlackbodyModel, self).__init__(inp, out, name=name)


class ChromaticityModel(Model):

    def __init__(self, name="ChromaticityCompass"):
        inp = Input((1, 6208, 5), name="SkyFeatures")
        out = Conv2D(10, (1, 5), data_format='channels_first', activation='relu', name="PixelConv10")(inp)
        self.filter = Model(inp, out, name="%s-filter" % name)

        out = Flatten(name="PixelHidden0")(out)
        out = Dense(500, activation='relu', name="PixelHidden1")(out)
        out = Dropout(0.5, name="Dropout1-0.5")(out)
        out = Dense(100, activation='relu', name="PixelHidden2")(out)
        out = Dense(100, activation='relu', name="PixelHidden3")(out)
        out = Dense(100, activation='relu', name="PixelHidden4")(out)
        out = Dropout(0.5, name="Dropout2-0.5")(out)
        out = Dense(8, activation='tanh', name="Compass")(out)

        super(ChromaticityModel, self).__init__(inp, out, name=name)