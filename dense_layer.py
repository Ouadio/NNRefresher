import numpy as np
from typing import List, Callable, TypedDict


LossFunc_t = Callable[[np.array, np.array, bool], np.array]
ActivationFunc_t = Callable[[float], float]


"""
Weight biases gradients
"""


class WB_gradient(TypedDict):
    w: np.array
    b: np.array


"""
High-level Neural Network Layer
"""


class Layer:
    _in_size: int
    _out_size: int

    _z: np.array
    _a: np.array

    _is_out_layer: bool

    _sigma_l: np.array

    _activation_func: ActivationFunc_t
    _activation: str

    def __init__(self, in_size: int, out_size: int, type: str = "", activation: str = "identity") -> None:
        self._in_size = in_size
        self._out_size = out_size
        self._is_out_layer = False
        self._sigma_l = np.zeros(shape=[out_size, 1])
        self._activation = activation
        assert (activation in __ACTIVAIONS__), \
            f"Invalid activation function., here are the supported activation functions : {__ACTIVAIONS__.keys()}"
        self._activation_func = __ACTIVAIONS__.get(activation)
        print(
            f"Layer constructed in = {in_size}, out = {out_size}. Activation : {self._activation}")

    def get_in_size(self) -> int:
        return self._in_size

    def get_out_size(self) -> int:
        return self._out_size

    def forward(self, input: np.array, cache: bool = True) -> None:
        pass

    def get_deriv_z(self) -> np.array:
        temp_z: np.array = __ACTIVAIONS_DERIVATIVES__.get(
            self._activation)(self._z)
        return temp_z.copy()

    # Updates the weights, doesn't compute the gradients
    def backward(self, out_grad: WB_gradient, lr: float = 0.1) -> None:
        pass


class Flatten(Layer):

    _in_shape: np.array

    def __init__(self, out_size: int, in_shape: np.array) -> None:
        super().__init__(in_size=out_size, out_size=out_size, type="flatten")
        self._in_shape = in_shape

    def forward(self, input: np.array, cache: bool) -> None:
        z_ = input.flatten()
        self._a = z_
        if cache:
            self._z = z_


# Activation functions

def relu(x) -> float:
    return np.maximum(0, x)


def elu(x, alpha: float = 1):
    return (np.where(x >= 0, x, alpha*(np.exp(x)-1)))


def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


def tanh(x) -> float:
    return np.tanh(x)


def softmax(x) -> float:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # sum over columns for each row


def identity(x) -> float:
    return x

# Activation functions derivatives


def relu_der(x) -> float:
    return np.where(x >= 0, float(1), float(0))


def elu_der(x, alpha: float = 1) -> float:
    return (np.where(x >= 0, 1, alpha*np.exp(x)))


def sigmoid_der(x) -> float:
    return sigmoid(x)*(1-sigmoid(x))


def tanh_der(x) -> float:
    return (1 - tanh(x)**2)


def identity_der(x) -> float:
    return np.ones_like(x)


def softmax_der(x) -> None:
    return


# Supported activation functions
__ACTIVAIONS__ = {
    "identity": identity,
    "relu": relu,
    "elu": elu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax
}

__ACTIVAIONS_DERIVATIVES__ = {
    "identity": identity_der,
    "relu": relu_der,
    "elu": elu_der,
    "sigmoid": sigmoid_der,
    "tanh": tanh_der,
    "softmax": softmax_der
}

"""
Dense Neural Network Layer.
"""


class Dense(Layer):

    # Weights & Biases
    _W: np.array
    _b: np.array

    def __init__(self,  out_size: int, in_size: int, activation: str) -> None:
        super().__init__(in_size, out_size, type="dense", activation=activation)
        self._W = np.random.uniform(low=-1, high=1, size=[in_size, out_size])
        self._b = np.random.uniform(low=-1, high=1, size=[1, out_size])

    def forward(self, input: np.array, cache: bool = False) -> None:
        assert (
            input.shape[1] == self._in_size and input.shape[0] > 0), f"Input feature \
                size {input.shape[1]} doesn't match expected previous layer size : ${self._in_size}"

        z = np.dot(input, self._W) + self._b
        self._a = self._activation_func(z)

        # Pre-activation only needed during training
        if cache:
            self._z = z.copy()

    def get_weigths(self) -> np.array:
        return self._W

    def get_bias(self) -> np.array:
        return self._b

    # Updates the weights, doesn't compute the gradients
    def backward(self, out_grad: WB_gradient, lr: float = 0.1) -> None:

        w_grad = out_grad.get("w")
        b_grad = out_grad.get("b")

        self._W = self._W - lr * w_grad
        self._b = self._b - lr * b_grad


# Loss functions
def mse(y_pred: np.array, y_label: np.array, avg: bool = False):
    if avg:
        return (np.mean((y_pred-y_label)**2, axis=0))
    else:
        return (y_pred-y_label)**2


def mae(y_pred: np.array, y_label: np.array, avg: bool = False):
    if avg:
        return (np.mean(np.abs(y_pred-y_label), axis=0))
    else:
        return np.abs(y_pred-y_label)

# Loss function derivatives


def mse_der(aL: np.array, y_label: np.array, avg: bool = False):
    return (aL - y_label)


def mae_der(aL: np.array, y_label: np.array, avg: bool = False):
    return np.where(aL - y_label > 0, 1, -1)


# Loss functions
_LOSS_FUNCS_ = {"mse": mse,
                "mae": mae}


_LOSS_FUNCS_DERIV_ = {"mse": mse_der,
                      "mae": mae_der}

GradientList_t = List[np.array]

"""
Sequential Neural Network model representation, wrapping layers and 
offering training, inference and evaluation functionalities.
"""


class Sequential:
    # Custom local type
    SequenceLayers_t = List[Layer]

    # Whether it is in training or inference mode
    _training: bool
    # Layers & Nodes
    _layers: SequenceLayers_t
    _n_layers: int
    _n_neurons: int
    # Model's input & output
    __in_size: int
    __out_size: int
    # Whether model got compiled
    __compiled: bool
    # Loss function
    __loss_func: LossFunc_t
    __loss: str
    # Normalization parameters
    _in_features_mean: np.array
    _in_features_std: np.array
    # Gradients
    _gradients: GradientList_t

    _w_gradients: GradientList_t

    def __init__(self, layers: SequenceLayers_t = [], loss: str = "mse") -> None:
        self._layers = layers
        self._n_layers = len(layers)
        self.__in_size = 0
        self.__out_size = 0
        self.__compiled = False
        self.__loss_func = _LOSS_FUNCS_.get(loss)
        self.__loss = loss
        self._gradients = list()
        self._w_gradients = list()
        self._b_gradients = list()
        self._in_features_mean = np.empty(0)
        self._in_features_std = np.empty(0)
        self._normalize = False

    def compile(self, loss: str = "mse", normalize: bool = False) -> None:
        assert (loss in _LOSS_FUNCS_), "Unsupported loss function"
        self.__loss_func = _LOSS_FUNCS_.get(loss)
        self.__loss = loss

        L: int = self._n_layers

        for i in range(L):
            current_layer = self._layers[i]
            current_out_size = current_layer.get_out_size()

            if i != (L-1):
                next_layer = self._layers[i+1]
                next_in_size = next_layer.get_in_size()

                assert (current_out_size == next_in_size), \
                    f"Consecutive layers sizes don't match -> {current_out_size}!={next_in_size}"

                if isinstance(current_layer, Dense):
                    # Softmax only supported as last layer
                    hidden_activ = current_layer._activation_func
                    assert (hidden_activ != __ACTIVAIONS__.get("softmax")), \
                        "Hidden layer activation cannot be softmax"

            self._w_gradients.append(np.zeros_like(current_layer._W))
            self._b_gradients.append(np.zeros_like(current_layer._b))

        self._gradients = [np.zeros(0)] * L

        # Model's general inpu/output
        self.__in_size = self._layers[0].get_in_size()
        self.__out_size = self._layers[L-1].get_out_size()
        self._layers[L-1]._is_out_layer = True

        # Normalization parameters
        self._normalize = normalize
        if normalize:
            self._in_features_mean = np.zeros(
                shape=[1, self._layers[0].get_in_size()])
            self._in_features_std = np.ones(
                shape=[1, self._layers[0].get_in_size()])

        self.__compiled = True

        print("Compilation successful!")

    def forward(self, X_data: np.array, cache: bool = True) -> None:
        assert self._is_compiled(), "Model not compiled yet!"
        assert (
            X_data.shape[1] == self.__in_size and X_data.shape[0] > 0), \
            f"Input feature size {X_data.shape[1]} doesn't match expected previous layer size : {self.__in_size}"
        a_ = X_data.copy()
        for l in self._layers:
            l.forward(input=a_, cache=cache)
            a_ = l._a.copy()

        # print("Forward pass complete")

    def backward(self, lr: float = 0.1) -> None:

        L: int = self._n_layers
        for li in range(L):
            self._layers[li].backward(
                out_grad={"w": self._w_gradients[li], "b": self._b_gradients[li]}, lr=lr)

        print("backward pass complete ! ")

    def _is_compiled(self) -> bool:
        return self.__compiled

    def evaluate(self, X_data: np.array, y_label: np.array, normalized: bool = False):
        # Normalize
        if not normalized:
            X_data_ = np.divide(np.subtract(
                X_data, self._in_features_mean), self._in_features_std)
        else:
            X_data_ = X_data
        # Forward pass
        return (self.__loss_func(self.predict(X_data_), y_label, avg=True))

    def predict(self, X_data: np.array):
        self.forward(X_data=X_data, cache=True)
        return (self._layers[self._n_layers - 1]._a)

    def train(self,  X_data: np.array, y_label: np.array, n_iterations: int = 1, minibatch: int = 1, shuffle: bool = True, lr: float = 0.1, min_loss: float = 0) -> None:

        # Total data size
        m: int = X_data.shape[0]
        assert (m == y_label.shape[0]
                ), "Training data input & output sizes don't match"

        L: int = self._n_layers

        # Normalize in the first pass
        if (self._normalize):
            self._in_features_mean = np.mean(X_data, axis=0)
            self._in_features_std = np.std(X_data, axis=0)

        # Last layer
        layer = self._layers[L-1]
        layer_activation = layer._activation

        loss_func_der = _LOSS_FUNCS_DERIV_.get(self.__loss)
        last_activation_der = __ACTIVAIONS_DERIVATIVES__.get(layer_activation)

        # if it's zero, use the whole data
        if minibatch == 0:
            minibatch = m

        m_batches = int(m/minibatch)

        if self._normalize:
            X_data_ = np.subtract(X_data, self._in_features_mean)
            X_data_ = np.divide(X_data_, self._in_features_std)
        else:
            X_data_ = X_data.copy()

        y_label_ = y_label.copy()

        loss_history = np.zeros(shape=[n_iterations*m_batches])

        # Epochs
        for epoch in range(n_iterations):
            print(f"epoch : {epoch}")

            if shuffle:
                shuffle_idx = np.random.choice(
                    a=np.arange(0, m, 1), size=m, replace=False)
                X_data_ = X_data_[shuffle_idx, :].copy()
                y_label_ = y_label_[shuffle_idx, :].copy()

            # Mini-batches
            for batch_idx in range(m_batches):
                X_data_batch_ = X_data_[
                    batch_idx*minibatch:(batch_idx+1)*minibatch, :]
                y_label_batch_ = y_label_[
                    batch_idx*minibatch:(batch_idx+1)*minibatch, :]

                # Forward pass
                self.forward(X_data=X_data_batch_, cache=True)

                # Computing gradients
                # 1. Last layer
                minibatch_gradient = np.multiply(loss_func_der(
                    aL=layer._a, y_label=y_label_batch_), last_activation_der(layer._z))
                self._gradients[L -
                                1] = minibatch_gradient
                # 2. Remaining layers backward
                for li in np.arange(L-2, -1, step=-1):
                    lhs = np.dot(
                        self._gradients[li+1], self._layers[li+1]._W.transpose())
                    self._gradients[li] = np.multiply(
                        lhs, self._layers[li].get_deriv_z())

                # 3. Wheights gradients
                for li in np.arange(L-1, 0, step=-1):
                    activation_grad = self._layers[li-1]._a.transpose()
                    self._w_gradients[li] = np.dot(
                        activation_grad, self._gradients[li]) / minibatch

                    self._b_gradients[li] = np.mean(
                        self._gradients[li], axis=0).reshape(1, -1)

                # First layer
                activation_grad = X_data_batch_.transpose()
                self._w_gradients[0] = np.dot(
                    activation_grad, self._gradients[0]) / minibatch
                self._b_gradients[0] = np.mean(
                    self._gradients[0], axis=0).reshape(1, -1)

                # Backward pass (Updating Weights & Biases)
                self.backward(lr=lr)

                error_epoch: float = self.evaluate(
                    X_data=X_data_batch_, y_label=y_label_batch_, normalized=True)
                print(f"MSE = {error_epoch}")
                loss_history[m_batches*epoch + batch_idx] = error_epoch
                # End of 1 minibatch
            error_step: float = self.evaluate(
                X_data=X_data_, y_label=y_label_, normalized=True)
            print(f"total error = {error_step}")
            if (error_step <= min_loss):
                print(
                    f"Minimal loss reached after {m_batches*epoch + batch_idx} iterations")
                loss_history = loss_history[0:m_batches*epoch + batch_idx]
                break
            # End of 1 whole data pass (epoch)

        return loss_history


if __name__ == "__main__":
    print("Dense Layer module")
