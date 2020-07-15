from keras.applications.resnet50 import ResNet50
from keras.regularizers import l2
from keras.layers import (
    Input, Flatten, Dense, Dropout, Lambda, concatenate, BatchNormalization, Activation, Reshape
)
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras import backend as K
from intention_net.DRC import DRCNet as DRCN
# Some config/hyperparameters for the network
INITIALIZER = "he_normal"
L2 = 1e-5
DROPOUT = 0.3


def ResNet():
    ResNetModel = ResNet50(weights="imagenet")
    # Get all layers and their names correspondingly into a dict
    layers_dict = dict([(layer.name, layer) for layer in ResNetModel.layers])
    input = ResNetModel.layers[0].input
    # Only need the output of the avg_pool because we only want the feature abstraction part
    output = layers_dict["avg_pool"].output
    return Model(inputs=[input], outputs=[output])


def ResDRCNet(D, N, filters_list, kernel_size_list, padding="same", img=Input((224,224,3))):
    # encoded_img = EncoderNet()(img)
    resnet = ResNet()
    encoded_img = resnet(img)
    reshaped_img = Reshape(target_shape=(16, 16, 8))(encoded_img)
    drcn = DRCN(D, N, filters_list, kernel_size_list, padding)
    drc_output = drcn(reshaped_img)
    flattened = Flatten()(drc_output)
    output = Dense(2048, kernel_initializer=INITIALIZER, kernel_regularizer=l2(L2), activation="relu")(flattened)
    return Model(inputs=[img], outputs=[output])

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

"""Return a fully connected model with the given length"""
def FCModel(input_length, output_length):
    input = Input(shape=(input_length,))
    out = Dense(output_length, kernel_initializer=INITIALIZER, kernel_regularizer=l2(L2))(input)
    output = Dropout(DROPOUT)(out)
    return Model(inputs=[input], outputs=[output])

"""use the input intention to filter out the corresponding control"""
def filter_control(args):
    outs, intention = args[:-1], args[-1]
    outs = K.concatenate(outs, axis=0)
    batch_size = K.shape(intention)[0]
    intention_idx = K.cast(K.argmax(intention), 'int32') * batch_size + K.arange(0, batch_size)
    #return outs[intention_idx, :]
    return K.gather(outs, intention_idx)



"""IntentionNet function, to be called to give us the entire model"""
def IntentionNet(mode, input_frame, D, N, num_control, num_intentions=-1, use_side_model=True):
    print(f"Intention Mode: {mode}, Input_frame(Multi-camera or Mono): {input_frame}")
    # # Use ResNet to abstract features of the input image
    # feat_abstract_model = ResNet()

    """Define inputs of the IntentionNet"""
    # Part 1: camera input process
    if input_frame != "MULTI":
        rgb_input = Input(shape=(224, 224, 3))
        # visual features of size 2048
        # feat_abstract_model = ResDRCNet(D=D, N=N, filters_list=[32, 32, 64], kernel_size_list=[(3, 3), (3, 3), (3, 3)], img=rgb_input)
        resnet = ResNet()
        encoded_img = resnet(rgb_input)
        reshaped_img = Reshape(target_shape=(16, 16, 8))(encoded_img)
        drcn = DRCN(D, N, filters_list=[32, 32, 64, 64], kernel_size_list=[(3, 3), (3, 3), (3, 3), (3, 3)], padding="same")
        drc_output = drcn(reshaped_img)
        flattened = Flatten()(drc_output)
        output = Dense(2048, kernel_initializer=INITIALIZER, kernel_regularizer=l2(L2), activation="relu")(flattened)
        rgb_feat = [output]

    else:
        rgbl_input = Input(shape=(224, 224, 3))
        rgbm_input = Input(shape=(224, 224, 3))
        rgbr_input = Input(shape=(224, 224, 3))

        # Use side models for side views, which means no shared weights, different ResNet models
        if use_side_model:
            feat_abstract_model_l = ResDRCNet(D=D, N=N, filters_list=[32, 32, 64], kernel_size_list=[(3, 3), (3, 3), (3, 3)], img=rgbl_input)
            feat_abstract_model = ResDRCNet(D=D, N=N, filters_list=[32, 32, 64], kernel_size_list=[(3, 3), (3, 3), (3, 3)], img=rgbm_input)
            feat_abstract_model_r = ResDRCNet(D=D, N=N, filters_list=[32, 32, 64], kernel_size_list=[(3, 3), (3, 3), (3, 3)], img=rgbr_input)

            rgbl_feat = feat_abstract_model_l(rgbl_input)
            rgbm_feat = feat_abstract_model(rgbm_input)
            rgbr_feat = feat_abstract_model_r(rgbr_input)
        else:
            # Use the same model to process abstraction, shared weights
            feat_abstract_model = ResDRCNet(D=D, N=N, filters_list=[32, 32, 64], kernel_size_list=[(3, 3), (3, 3), (3, 3)], img=rgbm_input)

            rgbl_feat = feat_abstract_model(rgbl_input)
            rgbm_feat = feat_abstract_model(rgbm_input)
            rgbr_feat = feat_abstract_model(rgbr_input)

        # Process three outputs to reshape them then put together into one single 1024 layer
        rgbl_feat = Dropout(DROPOUT)(rgbl_feat)
        rgbl_feat = Dense(512, kernel_initializer=INITIALIZER, kernel_regularizer=l2(L2), activation="relu")(
            rgbl_feat)

        rgbm_feat = Dropout(DROPOUT)(rgbm_feat)
        rgbm_feat = Dense(1024, kernel_initializer=INITIALIZER, kernel_regularizer=l2(L2), activation="relu")(
            rgbm_feat)

        rgbr_feat = Dropout(DROPOUT)(rgbr_feat)
        rgbr_feat = Dense(512, kernel_initializer=INITIALIZER, kernel_regularizer=l2(L2), activation="relu")(
            rgbr_feat)
        # visual features of size 2048 (not concatenate yet)
        rgb_feat = [rgbl_feat, rgbm_feat, rgbr_feat]

    # Part 2: Intention input process
    if mode == "DLM":
        assert (num_intentions != -1), "Number of Intentions must be bigger than one"
        dlm_intention_input = Input(shape=(num_intentions,))
        # fully connected ReLU to abstract features
        dlm_intention_feat = FCModel(input_length=num_intentions, output_length=64)(dlm_intention_input)

        """
        ## Use speed as also an input
        speed_input = Input(shape=(1,))
        speed_feat = FCModel(input_length=1, output_length=64)(speed_input)
        feat_concat = concatenate(rgb_feat + [dlm_intention_feat, speed_feat])
        """
        feat_concat = concatenate(rgb_feat + [dlm_intention_feat])

        # controls
        controls = []
        for i in range(num_intentions):
            # fc linear
            out = Dropout(DROPOUT)(feat_concat)
            out = Dense(num_control, kernel_initializer=INITIALIZER, kernel_regularizer=l2(L2))(out)
            controls.append(out)
        controls.append(dlm_intention_input)
        print("number of intentions: ", num_intentions)
        control = Lambda(filter_control, output_shape=(num_control,))(controls)

        if input_frame != "MULTI":
            intention_net_model = Model(inputs=[rgb_input, dlm_intention_input], outputs=control)
        else:
            intention_net_model = Model(inputs=[rgbl_input, rgbm_input, rgbr_input, dlm_intention_input], outputs=control)

    else:
        lpe_intention_input = Input(shape=(224, 224, 3))
        if mode == "LPE_SIAMESE":
            lpe_intention_feat = feat_abstract_model(lpe_intention_input)
        else:
            assert (mode == "LPE_NO_SIAMESE"), "LPE INTENTION MODE WITHOUT SIAMESE ARCHITECTURE"
            lpe_feat_abstract_model = ResDRCNet(D=D, N=N, filters_list=[32, 32, 64], kernel_size_list=[(3, 3), (3, 3), (3, 3)], img=lpe_intention_input)
            lpe_intention_feat = lpe_feat_abstract_model(lpe_intention_input)
        # Use speed as also an input
        speed_input = Input(shape=(1,))
        speed_feat = FCModel(input_length=1, output_length=64)(speed_input)

        feat_concat = concatenate(rgb_feat + [lpe_intention_feat, speed_feat])
        out = Dropout(DROPOUT)(feat_concat)
        out = Dense(2048, kernel_initializer=INITIALIZER, kernel_regularizer=l2(L2), activation="relu")(out)
        out = Dropout(DROPOUT)(out)
        control = Dense(num_control, kernel_initializer=INITIALIZER, kernel_regularizer=l2(L2))(out)

        if input_frame != "MULTI":
            intention_net_model = Model(inputs=[rgb_input, lpe_intention_input, speed_input], outputs=[control])
        else:
            intention_net_model = Model(inputs=[rgbl_input, rgbm_input, rgbr_input, lpe_intention_input, speed_input], outputs=[control])

    # finished building model
    return intention_net_model

if __name__ == "__main__":
    # do some test here
    from keras.preprocessing.image import load_img, img_to_array
    from keras.applications.resnet50 import preprocess_input
    from keras.utils import to_categorical
    import numpy as np
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    img = load_img("/home/davidzhuoran/intentionNet_zhuoran/intention_net/test/dog.jpeg", target_size=(224, 224))
    img = preprocess_input(img_to_array(img))
    img = np.expand_dims(img, axis=0) # expand dims to make it a batch

    intention_net = IntentionNet("DLM", input_frame="NORMAL", D=3, N=8, num_control=2, num_intentions=4)
    intention_net.load_weights("/home/davidzhuoran/intentionNet_zhuoran/intention_net/model_saved/ResDRC/NORMAL_DLM_D3_N6_latest_mode.h5")
    print(intention_net.summary())
    print(img.shape)
    control = intention_net.predict([img, to_categorical([1], num_classes=4)])
    print(control)












