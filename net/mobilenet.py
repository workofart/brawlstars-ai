from keras.applications import MobileNetV2
from keras import Model
from keras.layers import Dense, Flatten


def mnet():
    net = MobileNetV2()

    x = net.layers[-6].output
    x = Flatten()(x)
    predictions = Dense(6, activation='softmax')(x)

    # Input should be (None, 224, 224, 3)
    model = Model(inputs=net.input, outputs=predictions)
    return model

def mnet_feature():
    net = MobileNetV2()
    x = net.layers[-5].output

    model = Model(inputs=net.input, outputs=x)
    return model