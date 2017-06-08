import numpy as np
import yaml
import os
from keras.models import Model
from keras.callbacks import Callback, TensorBoard
from keras.layers import *
from keras.regularizers import *
from backend import *
from learn.whitening import transform as trans, pca, zca
__dir__ = os.path.dirname(os.path.realpath(__file__))
__data__ = __dir__ + "/../data/"
__params__ = __data__ + "params/"
__logs__ = __dir__ + "/../logs/"


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

    def __init__(self, inputs, outputs, data_shape=None, filters=None, name=None,
                 transform=None, optimizer="rmsprop", loss="mae", metrics=[],
                 epochs=100, batch_size=360, shuffle=False, reset_state=None, load_weights=False):
        super(CompassModel, self).__init__(inputs=inputs, outputs=outputs, name=name)
        if filters is not None:
            self.filters = filters
        self.data_shape = data_shape
        self.transform = transform
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reset_state = reset_state
        if load_weights:
            self.load_weights(__params__ + "%s.h5" % self.name)

    def train(self, train_data, valid_data=None):
        # compile the model
        self.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        # set training parameters
        kwargs = {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'shuffle': self.shuffle
        }

        # set and transform training data
        x, y = train_data
        if self.transform is not None:
            x = self.transform(x)
        x = x.reshape(self.data_shape)

        # set and transform validation data
        if valid_data is not None:
            x_test, y_test = valid_data
            if self.transform is not None:
                x_test = self.transform(x_test)
            kwargs['validation_data'] = (x_test.reshape(self.data_shape), y_test)

        # set callbacks
        kwargs['callbacks'] = [TensorBoard(log_dir=__logs__ + self.name)]
        if self.reset_state is not None:
            kwargs['callbacks'].append(ResetStatesCallback(self.reset_state))

        # train model
        hist = self.fit(x, y, **kwargs)

        # save the weights
        self.save_weights(__params__ + "%s.h5" % self.name, overwrite=True)

        return hist

    def predict(self, x, **kwargs):
        if 'batch_size' not in kwargs.keys():
            kwargs['batch_size'] = self.batch_size
        if self.transform is not None:
            x = self.transform(x)
        x = x.reshape(self.data_shape)
        return super(CompassModel, self).predict(x, **kwargs)

    @classmethod
    def load_dataset(cls, data, x_shape=(-1, 1, 6208, 5), y_shape=(-1, 8), time="full",
                     pol=True, directionwise=False, ret_reset_state=False):
        """
        
        :param data: the data to load. Can be a string denoting the path of the file to load, a dictionary with 'x' and
         'y' keys, or a tuple with the inputs in the first position and the outputs in the second one
        :param x_shape: the input shape
        :param y_shape: the output shape 
        :param pol: True to keep only the polarisation information
        :param directionwise: True to change the order of the data and sort the using the direction
        :param ret_reset_state: True to return the reset state indexes
        :return: x, y (, reset_state)
        """
        # BlackBody x_shape=(-1, 1, 104, 473)
        if x_shape[0] != -1:
            x_shape = (-1,) + x_shape
        pol = pol or x_shape[-1] == 2
        reset_state = []
        if isinstance(data, list) or isinstance(data, tuple):
            if isinstance(data[0], basestring):
                names = data
                x, y = [], []
                for name in names:
                    print "Loading '%s.npz' ..." % name,
                    src = np.load(__data__ + 'datasets/%s.npz' % name)
                    x0 = src['x']
                    y0 = src['y'][(x_shape[1]-1):]
                    if "full" in time:
                        pass
                    elif "morning" in time:
                        y0 = y0[:(x0.shape[0]/2)]
                        x0 = x0[:(x0.shape[0]/2)]
                    elif "afternoon" in time:
                        y0 = y0[(x0.shape[0]/2):]
                        x0 = x0[(x0.shape[0]/2):]
                    if len(x_shape) == 4 and x_shape[1] > 1:
                        x00 = []
                        for i in xrange(x_shape[1]):
                            x00.append(x0[i:x0.shape[0]-(x_shape[1]-i-1)])
                        x0 = np.array(x00).swapaxes(0, 1)
                    print x0.shape, y0.shape
                    x.append(x0)
                    y.append(y0)
                    reset_state.append(x[-1].shape[0] / 360)

                x = np.concatenate(tuple(x), axis=0)
                y = np.concatenate(tuple(y), axis=0)
            elif len(data) == 2:
                x, y = data
            else:
                x, y = np.zeros(x_shape), np.zeros(y_shape)
        elif isinstance(data, dict):
            x, y = data['x'], data['y']
        else:
            raise AttributeError("Unrecognised input data!")
        x = cls.__transform_input(x, x_shape, pol)
        y = cls.__transform_output(y, y_shape)

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

    @classmethod
    def __transform_input(cls, x, x_shape=(-1, 1, 6208, 5), pol=False):
        if pol and len(x.shape) > 2:
            x_shape = x_shape[:-1] + (2,)
            x = x[..., -2:]
        return x.reshape(x_shape)

    @classmethod
    def __transform_output(cls, y, y_shape=(-1, 8)):
        if len(y_shape) < 2:
            y_shape = (-1,) + y_shape
        if len(y.shape) > 1:
            if y.shape[-1] == 8:
                if y_shape[-1] == 8:
                    return y
                y = np.rad2deg(compass2rad(y))
            elif y.shape[-1] == 360:
                if y_shape[-1] == 360:
                    return y
                y = np.argmax(y, axis=-1)
            elif y.shape[-1] == 1:
                pass
            else:
                raise AttributeError("Unknown output shape: %s" % str(y_shape))

        if y_shape[-1] == 8:
            y = rad2compass(np.deg2rad(y))
        elif y_shape[-1] == 360:
            y = np.eye(360)[y]
        else:
            y = np.deg2rad(y)

        return y.reshape(y_shape)


def from_file(filename):
    params = __load_config__(filename)
    models = []
    layers = params.pop('layers')
    x = inp = layers[0]
    for i, layer in enumerate(layers):
        if i == 0:
            continue
        # layers[i] = _create_layer(**layer)
        x = layers[i](x)
        if layers[i].name in params['filters']:
            models.append(Model(inp, x, name="%s-%s" % (params['name'], layers[i].name)))
    params['filters'] = models
    if not ('data_shape' in params.keys()):
        print inp
        params['data_shape'] = (-1,) + tuple(inp.get_shape().as_list())[1:]
    return CompassModel(inp, x, **params)


def __load_config__(filename):
    with open(__data__ + "models/" + filename, 'r') as f:
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
            param = __eval_list__(param)
        elif isinstance(param, dict):
            param = __eval_dict__(param)
        else:
            param = eval(param)
    except NameError:
        pass
    except TypeError:
        pass
    return param


def __eval_list__(param):
    for l in xrange(len(param)):
        param[l] = __eval__(param[l])
    return param


def __eval_dict__(param):
    for k in param.keys():
        param[k] = __eval__(param[k])
        if 'layers' in k:
            for l in xrange(len(param[k])):
                param[k][l] = __create_layer(**param[k][l])
        elif 'regularizer' in k:
            r = param[k].keys()[-1]
            param[k] = __eval__(r)(param[k][r])
        elif 'transform' in k:
            kwargs = param[k].copy()
            param[k] = lambda x: trans(x, **kwargs)
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
