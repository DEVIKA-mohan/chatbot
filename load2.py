# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:50:59 2020

@author: DEVIKA
"""

import keras.models  
import tensorflow as tf   
from keras.models import model_from_json
from keras import backend as K

K.clear_session()
def init():

    json_file = open('Model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("Model/model.h5")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    
    graph = tf.compat.v1.get_default_graph
    
    return model,graph

init()



