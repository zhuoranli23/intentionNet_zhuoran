from keras import backend as K
from keras.layers import Layer
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout
from keras.regularizers import l2
import tensorflow as tf
from keras.models import Sequential, Model
import numpy as np
INITIALIZER = "he_normal"
L2 = 1e-5
DROPOUT = 0.3
"""
Use the functional API.

Define a custom layer where the call method accepts a list of tensors (and may return a list of tensors, 
or just a single tensor).
Use it like:
c, d = layer([a, b])
Note that the get_output_shape_for method should also accept a list of shape tuples.
def call(self, inputTensor):

    #calculations with inputTensor and the weights you defined in "build"
    #inputTensor may be a single tensor or a list of tensors

    #output can also be a single tensor or a list of tensors
    return [output1,output2,output3]
def compute_output_shape(self,inputShape):
    input_shape:  [(None, 4), (None, 4, 5)]
    #calculate shapes from input shape    
    return [shape1,shape2,shape3]
"""

class DRCCell(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="same", kernel_initializer="he_normal", **kwargs):
        super(DRCCell, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        """
        :param input_shape: A list of 4 tensors' shape tuples: [it, c(n-1)d, h(n-1)d, h(n)(d-1)]
        :return:
        """
        assert isinstance(input_shape, list), "input shape must be a list of tuples"

        # Calculate dimensions
        it_channels = input_shape[0][-1]
        prev_h_channels = input_shape[3][-1]
        self.it_kernel_shape = self.kernel_size + (it_channels, self.filters)
        self.prev_h_shape = self.kernel_size + (prev_h_channels, self.filters)
        self.h_c_shape = self.kernel_size + (self.filters, self.filters)
        self.c_shape = input_shape[1][1:]

        # Weight matrix corresponding to it
        self.Wfi = self.add_weight(shape=self.it_kernel_shape, initializer=self.kernel_initializer,
                                   name="Wfi", regularizer=l2(1e-4), trainable=True)
        self.Wii = self.add_weight(shape=self.it_kernel_shape, initializer=self.kernel_initializer,
                                   name="Wii", regularizer=l2(1e-4), trainable=True)
        self.Woi = self.add_weight(shape=self.it_kernel_shape, initializer=self.kernel_initializer,
                                   name="Woi", regularizer=l2(1e-4), trainable=True)
        self.Wci = self.add_weight(shape=self.it_kernel_shape, initializer=self.kernel_initializer,
                                   name="Wci", regularizer=l2(1e-4), trainable=True)

        # Weight matrix correspoding to h(n)(d-1)
        self.Wfh1 = self.add_weight(shape=self.prev_h_shape, initializer=self.kernel_initializer,
                                    name="Wfh1", regularizer=l2(1e-4), trainable=True)
        self.Wih1 = self.add_weight(shape=self.prev_h_shape, initializer=self.kernel_initializer,
                                    name="Wih1", regularizer=l2(1e-4), trainable=True)
        self.Woh1 = self.add_weight(shape=self.prev_h_shape, initializer=self.kernel_initializer,
                                    name="Woh1", regularizer=l2(1e-4), trainable=True)
        self.Wch1 = self.add_weight(shape=self.prev_h_shape, initializer=self.kernel_initializer,
                                    name="Wch1", regularizer=l2(1e-4), trainable=True)

        # Weight matrix correspoding to h(n-1)d
        self.Wfh2 = self.add_weight(shape=self.h_c_shape, initializer=self.kernel_initializer,
                                    name="Wfh2", regularizer=l2(1e-4), trainable=True)
        self.Wih2 = self.add_weight(shape=self.h_c_shape, initializer=self.kernel_initializer,
                                    name="Wih2", regularizer=l2(1e-4), trainable=True)
        self.Woh2 = self.add_weight(shape=self.h_c_shape, initializer=self.kernel_initializer,
                                    name="Woh2", regularizer=l2(1e-4), trainable=True)
        self.Wch2 = self.add_weight(shape=self.h_c_shape, initializer=self.kernel_initializer,
                                    name="Wch2", regularizer=l2(1e-4), trainable=True)

        # Weight matrix corresponding to c(n-1)d
        self.Wfc = self.add_weight(shape=self.c_shape, initializer=self.kernel_initializer,
                                   name="Wfc", regularizer=l2(1e-4), trainable=True)
        self.Wic = self.add_weight(shape=self.c_shape, initializer=self.kernel_initializer,
                                   name="Wic", regularizer=l2(1e-4), trainable=True)
        self.Woc = self.add_weight(shape=self.c_shape, initializer=self.kernel_initializer,
                                   name="Woc", regularizer=l2(1e-4), trainable=True)

        # Weight matrix for bias
        self.bias_f = self.add_weight(shape=(self.filters,), initializer=self.kernel_initializer,
                                      name="bias_f", regularizer=l2(1e-4), trainable=True)
        self.bias_i = self.add_weight(shape=(self.filters,), initializer=self.kernel_initializer,
                                      name="bias_i", regularizer=l2(1e-4), trainable=True)
        self.bias_c = self.add_weight(shape=(self.filters,), initializer=self.kernel_initializer,
                                      name="bias_c", regularizer=l2(1e-4), trainable=True)
        self.bias_o = self.add_weight(shape=(self.filters,), initializer=self.kernel_initializer,
                                      name="bias_o", regularizer=l2(1e-4), trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):
        """
        :param inputs: A list of 4 tensors: [it, c(n-1)d, h(n-1)d, h(n)(d-1)]
        :return:
        """
        assert isinstance(inputs, list), "input shape must be a list of tuples"

        def conv(x, w, padding="same"):
            return K.conv2d(x, w, strides=self.strides, padding=padding)

        it, c_n1_d, h_n1_d, h_n_d1 = inputs

        f = K.hard_sigmoid(conv(it, self.Wfi) + conv(h_n_d1, self.Wfh1) + conv(h_n1_d, self.Wfh2) + self.Wfc * c_n1_d + self.bias_f)
        i = K.hard_sigmoid(conv(it, self.Wii) + conv(h_n_d1, self.Wih1) + conv(h_n1_d, self.Wih2) + self.Wic * c_n1_d + self.bias_i)
        o = K.hard_sigmoid(conv(it, self.Woi) + conv(h_n_d1, self.Woh1) + conv(h_n1_d, self.Woh2) + self.Woc * c_n1_d + self.bias_o)
        c = f * c_n1_d + i * K.tanh(conv(it, self.Wci) + conv(h_n_d1, self.Wch1) + conv(h_n1_d, self.Wch2) + self.bias_c)
        h = o * K.tanh(c)
        return [h, c]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list), "input shape must be a list of tuples"
        return [input_shape[2], input_shape[1]]



class DRCNet(Layer):
    def __init__(self, D, N, filters_list, kernel_size_list, padding="same",**kwargs):
        super(DRCNet, self).__init__(**kwargs)
        self.depth = D
        self.steps = N
        self.filters_list = filters_list
        self.kernel_size_list = kernel_size_list
        # self.strides_list = strides_list
        self.padding = padding
        self.internal_states = []
        self.layers = {}
        self.hidden_states = {}

        for i in range(self.depth):
            cell_name = f"cell{i}"
            cell = DRCCell(self.filters_list[i], self.kernel_size_list[i])
            setattr(self, cell_name, cell)
            self.layers[cell_name] = cell

    def initialize_hidden_states(self, input_shape):
        for i in range(self.depth):
            cell_name = f"cell{i}"
            # h = K.zeros((1, input_shape[1], input_shape[2], self.filters_list[i]))
            # c = K.zeros((1, input_shape[1], input_shape[2], self.filters_list[i]))
            h = tf.placeholder_with_default(K.zeros((1, input_shape[1], input_shape[2], self.filters_list[i])),
                                            [None, input_shape[1], input_shape[2], self.filters_list[i]])
            c = tf.placeholder_with_default(K.zeros((1, input_shape[1], input_shape[2], self.filters_list[i])),
                                            [None, input_shape[1], input_shape[2], self.filters_list[i]])

            self.hidden_states[cell_name] = [h, c]

    def build(self, input_shape):
        # build weights of cells
        for i in range(self.depth):
            # [it, c(n - 1)d, h(n - 1)d, h(n)(d - 1)]
            cell_name = f"cell{i}"
            it_shape = input_shape
            c_n1_d_shape = (input_shape[0], input_shape[1], input_shape[2], self.filters_list[i])
            h_n1_d_shape = (input_shape[0], input_shape[1], input_shape[2], self.filters_list[i])
            if i == 0:
                h_n_d1_shape = (input_shape[0], input_shape[1], input_shape[2], self.filters_list[-1])
            else:
                h_n_d1_shape = (input_shape[0], input_shape[1], input_shape[2], self.filters_list[i-1])
            # build cell weight
            getattr(self, cell_name).build([it_shape, c_n1_d_shape, h_n1_d_shape, h_n_d1_shape])
            self.trainable_weights += getattr(self, cell_name).trainable_weights

        if len(self.hidden_states) == 0:
            self.initialize_hidden_states(input_shape)
        self.built = True


    def call(self, inputs, **kwargs):
        """
        :param inputs: initial input, it
        :return: final output, h_n_D
        """
        it = inputs

        for step in range(self.steps):
            for d in range(self.depth):
                cell_name = f"cell{d}"

                h_n1_d, c_n1_d = self.hidden_states[cell_name]
                if d == 0:
                    h_n_d1 = self.hidden_states[f"cell{self.depth-1}"][0]
                else:
                    h_n_d1 = self.hidden_states[f"cell{d-1}"][0]
                h, c = getattr(self, cell_name)([it, c_n1_d, h_n1_d, h_n_d1])

                self.hidden_states[cell_name] = [h, c]
        return self.hidden_states[f"cell{self.depth-1}"][0]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters_list[-1])

def DRCN(D, N, filters_list, kernel_size_list, padding="same", img=Input((224, 224, 3))):
    encoded_img = EncoderNet()(img)
    drc_output = DRCNet(D, N, filters_list, kernel_size_list, padding)(encoded_img)
    flattened = Flatten()(drc_output)
    output = Dense(2048, kernel_initializer=INITIALIZER, kernel_regularizer=l2(L2), activation="relu")(flattened)
    return Model(inputs=img, outputs=output)

def EncoderNet():
    img = Input(shape=(224, 224, 3))

    cnn1 = Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), padding="same", activation="relu",
                  kernel_initializer=INITIALIZER, kernel_regularizer=l2(1e-4))(img)
    cnn2 = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding="same", activation="relu",
                  kernel_initializer=INITIALIZER, kernel_regularizer=l2(1e-4))(cnn1)
    cnn3 = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", activation="relu",
                  kernel_initializer=INITIALIZER, kernel_regularizer=l2(1e-4))(cnn2)
    cnn4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu",
                  kernel_initializer=INITIALIZER, kernel_regularizer=l2(1e-4),)(cnn3)
    cnn5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu",
                  kernel_initializer=INITIALIZER, kernel_regularizer=l2(1e-4))(cnn4)
    cnn6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu",
                  kernel_initializer=INITIALIZER, kernel_regularizer=l2(1e-4))(cnn5)

    encoded_img = cnn6

    return Model(inputs=[img], outputs=[encoded_img])


if __name__ == "__main__":
    # do some test here
    from keras.preprocessing.image import load_img, img_to_array
    from keras.applications.resnet50 import preprocess_input
    from keras.utils import to_categorical
    import numpy as np

    img = load_img("/Users/davidlee/IntentionNet_Zhuoran/intention_net/test/dog.jpeg", target_size=(224, 224))
    img = preprocess_input(img_to_array(img))
    img = np.expand_dims(img, axis=0)

    # expand dims to make it a batch

    # it = Input((224, 224, 3))
    # drc = DRCNet(D=3, N=5, filters_list=[32, 32, 64], kernel_size_list=[(8, 8), (8, 8), (8, 8)])
    # out = drc(it)
    # model = Model(inputs=[it], outputs=[out])
    # print(model.predict([img]))
    # it = Input((48,))
    # fc = FCModel()
    # out = fc(it)
    # model = Model(inputs=[it], outputs=[out])
    # print(model.predict([np.random.random((1, 48))]))
    """
    Wrong Version: InvalidArgumentError: You must feed a value for placeholder tensor 'input_2' with dtype float 
    Reasons: DRCN has input tensor cannot get the value
    it = Input((224, 224, 3))
    feat_abstract_model = DRCN(D=3, N=5, filters_list=[32, 32, 32], kernel_size_list=[(3, 3), (3, 3), (3, 3)])
    out = feat_abstract_model(it)
    model = Model(inputs=it, outputs=out)
    print(model.predict([img]))
    """

    """
    Successful Version
    it = Input((224, 224, 3))
    feat_abstract_model = DRCN(D=3, N=5, filters_list=[32, 32, 32], kernel_size_list=[(3, 3), (3, 3), (3, 3)], img=it)
    out = feat_abstract_model(it)
    model = Model(inputs=it, outputs=out)
    print(model.summary())
    print(model.predict([img]).shape)
    """














