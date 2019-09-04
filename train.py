# organize imports
from __future__ import print_function

import matplotlib.pyplot as plt
import argparse
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
import time
import os
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
import keras.backend as K
import math

# from dense_inception import DenseNetInception
from models import DenseNetBaseModel, Inceptionv3Model, VGGModel, RestNetModel, DenseFoodModel

from utils import Params
from loss_history import LossHistory
from loss_fn import get_center_loss, get_softmax_loss, get_total_loss



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_dir", required=True,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model_dir", required=True,
                help="path to Model config (i.e., directory of Model)")
ap.add_argument("-r", "--restore_from", required=False,
                help="path of saved checkpoints (i.e., directory of check points)")

args = vars(ap.parse_args())

# Arguments
data_dir = args["data_dir"]
model_dir = args["model_dir"]
restore_from = args["restore_from"]

# load the user configs

params = Params(os.path.join(model_dir, 'params.json'))

# config variables
LAMBDA = 0.5
CENTER_LOSS_ALPHA = 0.5
LOSS_FN = params.loss_fn
EPOCHS = params.num_epochs
INIT_LR = params.learning_rate
BS = params.batch_size
num_layers_per_block = params.num_layers_per_block
INPUT1_DIMS = (params.image1_size, params.image1_size, 3)
INPUT2_DIMS = (params.image2_size, params.image2_size, 3)
seed      = 2019
model_name = params.model_name
use_imagenet_weights = params.use_imagenet_weights
save_period_step = params.save_period_step
num_inputs = params.num_inputs
history_filename = os.path.join(model_dir, "train_fit_history.json")

# Dataset Directory
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "eval")

train_datagen = ImageDataGenerator(rotation_range=25,
                                   width_shift_range=0.1,
                                   rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)


single_train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(INPUT1_DIMS[1], INPUT1_DIMS[1]),
        batch_size=BS,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
single_validation_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=(INPUT1_DIMS[1], INPUT1_DIMS[1]),
        batch_size=BS,
        class_mode='categorical')

CLASSES = single_train_generator.num_classes
params.num_labels = CLASSES

# initialize the model
print("[INFO] creating model...")
overwriting = os.path.exists(history_filename) and restore_from is None
assert not overwriting, "Weights found in model_dir, aborting to avoid overwrite"
loss_history = LossHistory(history_filename)
EPOCHS += loss_history.get_initial_epoch()

if LOSS_FN == 'center':
    loss_fn = get_center_loss(CENTER_LOSS_ALPHA, CLASSES)
elif LOSS_FN == 'softmax':
    loss_fn = get_softmax_loss()
else:
    loss_fn = get_total_loss(LAMBDA, CENTER_LOSS_ALPHA, CLASSES)

if restore_from is None:
    if model_name == 'densenet':
        model = DenseNetBaseModel(CLASSES, use_imagenet_weights).model
    elif model_name == "inception":
        model = Inceptionv3Model(CLASSES, use_imagenet_weights).model
    elif model_name == 'vgg':
        model = VGGModel(CLASSES, use_imagenet_weights).model
    elif model_name == "resnet":
        model = RestNetModel(CLASSES, use_imagenet_weights).model
    elif model_name == "densefood":
        model = DenseFoodModel(CLASSES, num_layers_per_block).model
    else:
        assert False, "Cannot find the model name on the list"
else:
    # Restore Model
    file_path = os.path.join(restore_from)
    assert os.path.exists(file_path), "No model in restore from directory"
    model = load_model(file_path, custom_objects={'loss_fn': loss_fn})


# Initial checkpoints and Tensorboard to monitor training

print("[INFO] compiling model...")


def cosine_decay(epoch):
    initial_lrate = INIT_LR
    lrate = 0.5 * initial_lrate * (1 + math.cos(epoch*math.pi/EPOCHS))
    return lrate

def step_decay(epoch):
    initial_lrate = INIT_LR
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

if params.decay == 'step':
    lrate = LearningRateScheduler(step_decay)
else:
    lrate = LearningRateScheduler(cosine_decay)

opt = Adam(lr=0.0, decay=0.0)
# opt = SGD(INIT_LR) #Adam(lr=INIT_LR)
model.compile(loss=loss_fn, optimizer=opt,
              metrics=["accuracy", "top_k_categorical_accuracy"])


tensorBoard = TensorBoard(log_dir=os.path.join(model_dir, 'logs/{}'.format(time.time())), write_images=True)

best_checkpoint = ModelCheckpoint(os.path.join(model_dir, "best.weights.hdf5"),
                                  monitor='val_acc',
                                  period=save_period_step,
                                  verbose=1, save_best_only=True, mode='max')
last_checkpoint = ModelCheckpoint(os.path.join(model_dir, "last.weights.hdf5"),
                                  monitor='val_acc',
                                  period=save_period_step,
                                  verbose=1, mode='max')

print("[INFO] training started...")

history = model.fit_generator(
        single_train_generator,
        steps_per_epoch=single_train_generator.n // BS,
        initial_epoch=loss_history.get_initial_epoch(),
        epochs=EPOCHS,
        validation_data=single_validation_generator,
        validation_steps=single_validation_generator.n // BS,
        callbacks=[best_checkpoint, last_checkpoint, loss_history, lrate])


# save the model to disk
print("Saved model to disk")
model.save(os.path.join(model_dir, "last.weights.hdf5"))
