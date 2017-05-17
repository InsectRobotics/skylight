import numpy as np
import yaml
import os
from keras.models import Model
from keras.callbacks import Callback
from keras.layers import Dense, Conv2D, Input, Dropout, Flatten, LSTM, Reshape
from backend import rad2compass
__dir__ = os.path.dirname(os.path.realpath(__file__))


class ResetStatesCallback(Callback):
    def __init__(self, max_len=30):
        self.counter = 0
        self.max_len = max_len

    def on_batch_begin(self, batch, logs={}):
        if self.counter % self.max_len == 0:
            self.model.reset_states()
        self.counter += 1


class CompassModel(Model):

    def __init__(self, inputs, outputs, filters=None, name=None):
        super(CompassModel, self).__init__(inputs=inputs, outputs=outputs, name=name)
        if filters is not None:
            self.filters = filters

    def train(self, train_data, valid_data=None, batch_size=1, nb_epoch=100, shuffle=False, reset_state=30):
        x, y = self._load_dataset(train_data)
        kwargs = {
            'batch_size': batch_size,
            'nb_epoch': nb_epoch,
            'shuffle': shuffle
        }
        if valid_data is not None:
            x_test, y_test = self._load_dataset(valid_data)
            kwargs['validation_data'] = (x_test, y_test)

        loss, acc = self.fit(x, y, callbacks=[ResetStatesCallback(reset_state)], **kwargs)
        self.save_weights(__dir__ + "/../data/%s.h5" % self.name, overwrite=True)

        return loss, acc

    def test(self, data, batch_size=1):
        x, y = self._load_dataset(data)

        x = np.concatenate(tuple(x), axis=0)
        y = np.concatenate(tuple(y), axis=0)

        return self.evaluate(x, y, batch_size=batch_size)

    @classmethod
    def _load_dataset(cls, data, x_shape=(-1, 1, 6208, 5), directionwise=False):
        # BlackBody x_shape=(-1, 1, 104, 473)
        if isinstance(data, list) or isinstance(data, tuple):
            if isinstance(data[0], basestring):
                names = data
                x, y = [], []
                for name in names:
                    print "Loading '%s.npz' ..." % name
                    src = np.load(__dir__ + '/../data/%s.npz' % name)
                    x.append(src['x'].reshape(x_shape))
                    y.append(rad2compass(np.deg2rad(src['y'])))

                x = np.concatenate(tuple(x), axis=0)
                y = np.concatenate(tuple(y), axis=0)
            elif len(data) == 2:
                x, y = data
            else:
                x, y = np.zeros(x_shape)
        elif isinstance(data, dict):
            x, y = data['x'].reshape(x_shape), rad2compass(np.deg2rad(data['y']))
        else:
            raise AttributeError("Unrecognised input data!")

        if directionwise:
            x_, y_ = [], []
            for a in xrange(360):
                x_.append(x[a::360])
                y_.append(y[a::360])
            x = np.concatenate(tuple(x_), axis=0)
            y = np.concatenate(tuple(y_), axis=0)

        return x, y


def from_file(filename):
    with open(__dir__ + "/../data/" + filename, 'r') as f:
        try:
            params = yaml.load(f)
        except yaml.YAMLError as exc:
            print exc

    models = []
    layers = params['layers']
    x = inp = _create_layer(**layers[0])
    for i, layer in enumerate(layers[1:]):
        layers[i] = _create_layer(**layer)
        x = layers[i](x)
        if layers[i].name in params['filters']:
            models.append(Model(inp, x, name="%s-%s" % (params['name'], layers[i].name)))

    return CompassModel(inp, x, name=params['name'], filters=models)


def _create_layer(**kwargs):
    if not ('class' in kwargs.keys()):
        raise AttributeError("No 'class' key found. Unable to create layer!")

    c = eval(kwargs.pop('class'))
    for k in kwargs.keys():
        try:
            kwargs[k] = eval(kwargs[k])
        except NameError:
            pass
        except TypeError:
            pass
    return c(**kwargs)


if __name__ == "__main__":
    names = ["seville-cr-32-20170621", "seville-cr-32-20170601"]

    cr = from_file("chrom-mon-rnn.yaml")
    cr.summary()

    cr.train([names[0]], valid_data=[names[1]])
