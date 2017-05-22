import numpy as np
import yaml
import os
from keras.models import Model
from keras.callbacks import Callback, TensorBoard
from keras.layers import *
from backend import *
__dir__ = os.path.dirname(os.path.realpath(__file__))


class ResetStatesCallback(Callback):
    def __init__(self, reset_state=[30]):
        self.state = 0
        self.counter = 0
        self.reset_state = reset_state if isinstance(reset_state, list) else [reset_state]
        self.max_len = self.reset_state[0]

    def on_batch_begin(self, batch, logs={}):
        if self.counter % self.max_len == 0:
            self.model.reset_states()
        if self.counter // self.max_len > 0:
            self.state = (self.state + 1) % len(self.reset_state)
            self.max_len += self.reset_state[self.state]
            # print "\nHEY!", self.state, self.reset_state[self.state], self.max_len
        self.counter += 1


class CompassModel(Model):

    def __init__(self, inputs, outputs, data_shape=None, filters=None, name=None):
        super(CompassModel, self).__init__(inputs=inputs, outputs=outputs, name=name)
        if filters is not None:
            self.filters = filters
        self.data_shape = data_shape

    def train(self, train_data, valid_data=None, batch_size=360, nb_epoch=100, shuffle=False, reset_state=None):
        x, y = self.load_dataset(train_data)
        kwargs = {
            'batch_size': batch_size,
            'nb_epoch': nb_epoch,
            'shuffle': shuffle
        }
        if valid_data is not None:
            x_test, y_test = self.load_dataset(valid_data)
            kwargs['validation_data'] = (x_test, y_test)

        kwargs['callbacks'] = [TensorBoard(log_dir=__dir__ + "/../logs")]
        if reset_state is not None:
            kwargs['callbacks'].append(ResetStatesCallback(reset_state))
        hist = self.fit(x, y, **kwargs)
        self.save_weights(__dir__ + "/../data/%s.h5" % self.name, overwrite=True)

        return hist

    def test(self, data, batch_size=360):
        x, y = self.load_dataset(data)

        x = np.concatenate(tuple(x), axis=0)
        y = np.concatenate(tuple(y), axis=0)

        return self.evaluate(x, y, batch_size=batch_size)

    @classmethod
    def load_dataset(cls, data, x_shape=(-1, 1, 6208, 5), cx=True,
                     pol=True, directionwise=False, ret_reset_state=False):
        # BlackBody x_shape=(-1, 1, 104, 473)
        reset_state = []
        if pol:
            x_shape = x_shape[:-1] + (2,)
        if isinstance(data, list) or isinstance(data, tuple):
            if isinstance(data[0], basestring):
                names = data
                x, y = [], []
                for name in names:
                    print "Loading '%s.npz' ..." % name
                    src = np.load(__dir__ + '/../data/%s.npz' % name)
                    x0, y0 = src['x'], np.deg2rad(src['y'])
                    if pol:
                        x0 = x0[..., -2:]
                    if cx:
                        y0 = rad2compass(y0)
                    x.append(x0.reshape(x_shape))
                    y.append(y0)
                    reset_state.append(x[-1].shape[0] / 360)

                x = np.concatenate(tuple(x), axis=0)
                y = np.concatenate(tuple(y), axis=0)
            elif len(data) == 2:
                x, y = data
                if pol and len(x.shape) > 2:
                    x = x[..., -2:]
                if cx and ((len(y.shape) > 1 and y.shape[-1] == 1) or len(y.shape) <= 1):
                    y = rad2compass(y)
            else:
                x, y = np.zeros(x_shape), np.zeros(x_shape[0])
        elif isinstance(data, dict):
            x0, y0 = data['x'], np.deg2rad(data['y'])
            if pol:
                x0 = data['x'][..., -2:]
            if cx:
                y0 = rad2compass(y0)
            x, y = x0.reshape(x_shape), y0
        else:
            raise AttributeError("Unrecognised input data!")

        if directionwise:
            x_, y_ = [], []
            for a in xrange(360):
                x_.append(x[a::360])
                y_.append(y[a::360])
            x = np.concatenate(tuple(x_), axis=0)
            y = np.concatenate(tuple(y_), axis=0)

        if ret_reset_state:
            return x, y, reset_state
        else:
            return x, y


def from_file(filename):
    params = __load_config__(filename)
    models = []
    layers = params['layers']
    x = inp = layers[0]
    for i, layer in enumerate(layers):
        if i == 0:
            continue
        # layers[i] = _create_layer(**layer)
        x = layers[i](x)
        if layers[i].name in params['filters']:
            models.append(Model(inp, x, name="%s-%s" % (params['name'], layers[i].name)))

    if not ('data_shape' in params.keys()):
        print inp
        params['data_shape'] = (-1,) + tuple(inp.get_shape().as_list())[1:]
    return CompassModel(inp, x, data_shape=params['data_shape'], name=params['name'], filters=models)


def __load_config__(filename):
    with open(__dir__ + "/../data/" + filename, 'r') as f:
        try:
            params = yaml.load(f)
        except yaml.YAMLError as exc:
            print exc

    params = __eval__(params)
    # print params
    return params


def __eval__(param):
    try:
        if isinstance(param, list):
            for l in xrange(len(param)):
                param[l] = __eval__(param[l])
        elif isinstance(param, dict):
            for k in param.keys():
                param[k] = __eval__(param[k])
                if 'layers' in k:
                    for l, layer in enumerate(param[k]):
                        param[k][l] = __create_layer(**layer)
        else:
            param = eval(param)
    except NameError:
        pass
    except TypeError:
        pass
    return param


def __create_layer(**kwargs):
    if not ('class' in kwargs.keys()):
        raise AttributeError("No 'class' key found. Unable to create layer!")

    c = kwargs.pop('class')
    return c(**kwargs)


if __name__ == "__main__":
    names = ["seville-cr-32-20170621", "seville-cr-32-20170601"]

    cr = from_file("chrom-mon-rnn.yaml")
    cr.summary()

    cr.train([names[0]], valid_data=[names[1]])
