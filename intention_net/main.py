from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags
import numpy as np
import os
from tqdm import tqdm
from time import time

# keras import
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam, SGD
from keras_radam import RAdam
from keras.callbacks import TensorBoard
from keras.utils.training_utils import multi_gpu_model

from config import *
from intention_net_ResDRC import IntentionNet

import warnings

cfg = None
flags_obj = flags.FLAGS


class MyModelCheckpoint(Callback):
    """Save the model after every epoch.
        `filepath` can contain named formatting options,
        which will be filled the value of `epoch` and
        keys in `logs` (passed in `on_epoch_end`).
        For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
        then the model checkpoints will be saved with the epoch number and
        the validation loss in the filename.
        # Arguments
            filepath: string, path to save the model file.
            monitor: quantity to monitor.
            verbose: verbosity mode, 0 or 1.
            save_best_only: if `save_best_only=True`,
                the latest best model according to
                the quantity monitored will not be overwritten.
            mode: one of {auto, min, max}.
                If `save_best_only=True`, the decision
                to overwrite the current save file is made
                based on either the maximization or the
                minimization of the monitored quantity. For `val_acc`,
                this should be `max`, for `val_loss` this should
                be `min`, etc. In `auto` mode, the direction is
                automatically inferred from the name of the monitored quantity.
            save_weights_only: if True, then only the model's weights will be
                saved (`model.save_weights(filepath)`), else the full model
                is saved (`model.save(filepath)`).
            period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, bestpath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, skip=1):
        super(MyModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.bestpath = bestpath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.period = period
        self.skip = skip
        self.skip_count = 0
        self.epochs_since_last_save = 0

        if self.mode not in ["auto", "min", "max"]:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            self.mode = "auto"

        if self.mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif self.mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # logs will be filled in the method on_epoch_end at the end of the epoch
        logs = logs or {}
        self.epochs_since_last_save += 1
        self.skip_count += 1
        if flags_obj.num_gpus > 1:
            # discard multi_gpu layer
            old_model = self.model.layers[-2]
        else:
            old_model = self.model

        # Check whether reach the period that we should check to save model or not
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.bestpath

            # Only save the best
            if self.save_best_only:
                curr_quantity = logs.get(self.monitor)
                if curr_quantity is None:
                    warnings.warn("The monitored quantity %s is not available... Skipping..."
                                  % (self.monitor), RuntimeWarning)
                else:
                    # The Current one is better/the best
                    if self.monitor_op(curr_quantity, self.best):
                        # Whether keep silent or print out
                        if self.verbose > 0:
                            print("Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s !!!"
                                  % (epoch, self.monitor, self.best, curr_quantity, filepath))
                        # Save model/weights
                        if self.save_weights_only:
                            old_model.save_weights(filepath, overwrite=True)
                        else:
                            old_model.save(filepath, overwrite=True)
                        self.best = curr_quantity

                    # The Current one is NOT better
                    else:
                        if self.verbose > 0:
                            print("Epo %05d: %s did not improve" % (epoch, self.monitor))

        if self.skip_count >= self.skip:
            self.skip_count = 0
            filepath = self.filepath
            if self.verbose > 0:
                print("Epoch %05d: saving model to %s ..." % (epoch, filepath))
            if self.save_weights_only:
                old_model.save_weights(filepath, overwrite=True)
            else:
                old_model.save(filepath, overwrite=True)


def define_intention_net_flags():
    flags.DEFINE_enum(
        name="intention_mode", short_name="mode", default="DLM",
        enum_values=['DLM', 'LPE_SIAMESE', 'LPE_NO_SIAMESE'],
        help=help_wrap("Intention Net mode to run")
    )

    flags.DEFINE_enum(
        name="input_frame", short_name="input_frame", default="NORMAL",
        enum_values=["NORMAL", "WIDE", "MULTI"],
        help=help_wrap("Camera usage frame")
    )

    flags.DEFINE_enum(
        name="dataset", short_name="ds", default="PIONEER",
        enum_values=["PIONEER", "CARLA", "CARLA_SIM", "HUAWEI"],
        help=help_wrap("dataset to load for training")
    )

    global cfg
    # Get dict class._C
    cfg = load_config(IntentionNetConfig)
    """
    _C = EasyDict()
    _C.NUM_INTENTIONS = 4
    _C.WEIGHT_DECAY = 5e-5
    _C.MOMENTUM = 0.9
    """


def lr_schedule(epoch):
    """
    Learning rate schedule:
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    """
    lr = flags_obj.learning_rate
    if epoch > 90:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 5e-2
    elif epoch > 40:
        lr *= 1e-1
    elif epoch > 20:
        lr *= 5e-1
    print('Learning rate: ', lr)
    return lr


def get_optimizer():
    if flags_obj.optim == "rmsprop":
        optimizer = RMSprop(lr=flags_obj.learning_rate, decay=cfg.WEIGHT_DECAY, rho=0.9, epsilon=1e-08)
        print('=> use rmsprop optimizer')
    elif flags_obj.optim == "sgd":
        optimizer = SGD(lr=flags_obj.learning_rate, decay=cfg.WEIGHT_DECAY, momentum=cfg.MOMENTUM)
        print('=> use sgd optimizer')
    elif flags_obj.optim == "radam":
        optimizer = RAdam()
        print('=> use RAdam optimizer')
    else:
        optimizer = Adam(lr=flags_obj.learning_rate, decay=cfg.WEIGHT_DECAY)
        print('=> use adam optimizer')
    return optimizer


def main(_):
    global cfg
    global flags_obj
    cfg = load_config(IntentionNetConfig)
    flags_obj = flags.FLAGS
    ###################################
    # TensorFlow wizardry
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Only allow a total of half the GPU memory to be allocated
    # config.gpu_options.per_process_gpu_memory_fraction = 0.98
    # select which GPU to use for training
    config.gpu_options.visible_device_list = flags_obj.which_gpu #'1', '0,1,2'
    # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tf.Session(config=config))
    ###################################

    # Get validity data settled
    if flags_obj.val_dir is None:
        flags_obj.val_dir = flags_obj.data_dir

    # Get number of gpus, -1 means to use all gpus
    if flags_obj.num_gpus == -1:
        from tensorflow.python.client import device_lib
        # Get all local devices
        local_device_protos = device_lib.list_local_devices()
        flags_obj.num_gpus = sum([1 for device in local_device_protos if device.device_type == "GPU"])
    print("=> Using {} gpus".format(flags_obj.num_gpus))

    # Set the dataset to use
    if flags_obj.dataset == "PIONEER":
        from dataset import PioneerDataset as Dataset
        print('=> using pioneer data')
    elif flags_obj.dataset == "CARLA":
        from dataset import CarlaImageDataset as Dataset
        print('=> using CARLA published data')
    elif flags_obj.dataset == "CARLA_SIM":
        from dataset import CarlaSimDataset as Dataset
        print('=> using self-collected CARLA data')
    else:
        from dataset import HuaWeiFinalDataset as Dataset
        print('=> using HUAWEI data')

    # Set the model
    print('mode: ', flags_obj.mode, 'input frame: ', flags_obj.input_frame, 'batch_size', flags_obj.batch_size,
          "num_intentions: ", cfg.NUM_INTENTIONS)
    depth = flags_obj.DRC_depth
    steps = flags_obj.DRC_steps
    intention_net_model = IntentionNet(mode=flags_obj.intention_mode, input_frame=flags_obj.input_frame, D=depth,
                                       N=steps, num_control=Dataset.NUM_CONTROL, num_intentions=cfg.NUM_INTENTIONS)

    # Use multi-gpu to do parallel training
    #if flags_obj.num_gpus > 1:
        # make the model parallel
    #   flags_obj.batch_size = flags_obj.batch_size * flags_obj.num_gpus
    #    intention_net_model = multi_gpu_model(intention_net_model, flags_obj.num_gpus)

    # Print model summary
    print(intention_net_model.summary())

    # Optionally resume from a checkpoint
    if flags_obj.resume is not None:
        # Check whether it is a valid dir
        if os.path.isfile(flags_obj.resume):
            intention_net_model.load_weights(flags_obj.resume)
            print('=> loaded checkpoint {}'.format(flags_obj.resume))
            print("=> loaded successfully!!!")
        else:
            print("=> no checkpoint found at {}".format(flags_obj.resume))

    # Create Callbacks about learning rate
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=1e-7)

    # Set paths, save model, set tensorboard
    best_model_fn = os.path.join(flags_obj.model_dir, flags_obj.input_frame + "_" + flags_obj.intention_mode
                                 + f"_D{depth}_N{steps}" + "_best_mode.h5")
    latest_model_fn = os.path.join(flags_obj.model_dir, flags_obj.input_frame + "_" + flags_obj.intention_mode
                                 + f"_D{depth}_N{steps}" + "_latest_mode.h5")
    saveBestAndLatestModel = MyModelCheckpoint(latest_model_fn, best_model_fn, monitor="val_loss", verbose=1,
                                               save_best_only=True, save_weights_only=True, mode="auto", skip=10)
    tensorboard = TensorBoard(log_dir=f"logs/ResDRC/D{depth}_N{steps}"+"_{}".format(time()), write_graph=True, write_images=False,
                              batch_size=flags_obj.batch_size)

    # Get all Callbacks
    callbacks = [lr_scheduler, lr_reducer, saveBestAndLatestModel, tensorboard]

    # We choose max_samples to save time for training (since some dataset is very big).
    # For large dataset, we only sample 200000 samples every epoch
    training_generator = Dataset(flags_obj.data_dir, flags_obj.batch_size, cfg.NUM_INTENTIONS, mode=flags_obj.intention_mode,
                                 shuffle=False, max_samples=32000, input_frame=flags_obj.input_frame)
    validation_generator = Dataset(flags_obj.val_dir, flags_obj.batch_size, cfg.NUM_INTENTIONS, mode=flags_obj.intention_mode,
                                   shuffle=False, max_samples=1000, input_frame=flags_obj.input_frame)
    # Get optimizer
    optimizer = get_optimizer()

    # Model Compile and Model Fit
    intention_net_model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy", "mae"])

    intention_net_model.fit_generator(generator=training_generator, validation_data=validation_generator,
                                      use_multiprocessing=False, workers=flags_obj.num_workers, callbacks=callbacks,
                                      epochs=flags_obj.train_epochs)

if __name__ == "__main__":
    import tensorflow as tf

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    define_intention_net_flags()
    absl_app.run(main)
