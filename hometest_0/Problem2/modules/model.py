import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
    BatchNormalization,
    Conv1D,
    MaxPooling1D,
    Concatenate
)

def EEGModel(eeg_filters=32, eeg_dense=128, eye_filters=32 ,eye_dense=64):
    eeg_input = Input(shape=(64,1440))
    x = BatchNormalization()(eeg_input)
    x = Conv1D(eeg_filters, 3, activation="relu", input_shape=x.shape[1:])(x)
    x = MaxPooling1D( pool_size=2, strides=2, data_format="channels_last")(x)
    x = Flatten()(x)
    x = Dense(eeg_dense, activation="relu")(x)
    eeg_model = Model(inputs=eeg_input, outputs=x)


    eye_input = Input(shape=(64,31))
    y = BatchNormalization()(eye_input)
    y = Conv1D(eye_filters, 3, activation="relu", input_shape=x.shape[1:])(y)
    y = MaxPooling1D( pool_size=2, strides=2, data_format="channels_last")(y)
    y = Flatten()(y)
    y = Dense(eye_dense, activation='relu')(y)
    eye_model = Model(inputs=eye_input, outputs=y)

    combined = Concatenate()([eeg_model.output, eye_model.output])

    z = Dense(10, activation="relu")(combined)
    out = Dense(4, activation='softmax')(z)

    model = Model(inputs=[eeg_model.input, eye_model.input], outputs=out)

    return model

