#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:43:59 2022

@author: ru79seh
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd 
from matplotlib import pyplot as plt
import random
import tensorflow as tf
import pickle
import os
import pyreadr
import h5py

#%%
print(tf.test.is_built_with_cuda())
tf.config.list_physical_devices('GPU') 
#os.chdir('/~/CNN_for_conflict_pred')
#os.chdir('../dss/dsshome1/lxc04/ru79seh3/CNNs_for_conflict_pred')
#os.chdir('Syria/CNN')
#os.walk("../")

#tabular_df = pd.read_csv("data/cnn_est_df.csv", index_col = 0)
#full_matrix = np.load("data/full_matrix.npy")

tabular_df = pyreadr.read_r('D:/consulting/code_consulting/Africa/data/est_df.rds')
tabular_df = tabular_df[None]

matrix_path = 'D:/consulting/code_consulting/Africa/data/full_matrix.h5'
with h5py.File(matrix_path, 'r') as f:
    full_matrix = np.array(f['full_matrix'])

full_matrix = np.transpose(full_matrix, (4, 3, 2, 1, 0))
#%%
#syria example matrix
#full_matrix = np.load("D:/consulting/code_consulting/Africa/data/full_matrix.npy")

#%%
#tabular prep
train_tabular = tabular_df[(tabular_df["year"]<=2019)]
test_tabular = tabular_df[(tabular_df["year"]>2019)]

train_tabular_x = train_tabular[["xcoord", "ycoord", "intersect_area", 
                                 "ucdp_deaths_12_lag_1", "ucdp_deaths_12_lag_12", "ucdp_deaths_12_lag_24", "ucdp_deaths_12_neighbour_lag_1", "ucdp_deaths_12_neighbour_lag_12", "ucdp_deaths_12_neighbour_lag_24", 
           ]].to_numpy()
train_y = train_tabular[["ucdp_12_bin"]].to_numpy()[:,0]


test_tabular_x = test_tabular[["xcoord", "ycoord", "intersect_area", 
                                 "ucdp_deaths_12_lag_1", "ucdp_deaths_12_lag_12", "ucdp_deaths_12_lag_24", "ucdp_deaths_12_neighbour_lag_1", "ucdp_deaths_12_neighbour_lag_12", "ucdp_deaths_12_neighbour_lag_24", 
           ]].to_numpy()

test_y = test_tabular[["ucdp_12_bin"]].to_numpy()[:,0]

#%%

#matrix prep
train_matrix_x = full_matrix
print(train_matrix_x.shape)
test_matrix_x = full_matrix

train_matrix_x = np.repeat(full_matrix[:, :,np.newaxis, :, :, :], train_tabular_x.shape[0]/full_matrix.shape[1], axis=1)
print(train_matrix_x.shape)
test_matrix_x = np.repeat(full_matrix[:, :,np.newaxis, :, :, :], test_tabular_x.shape[0]/full_matrix.shape[1], axis=1)

#%%
train_matrix_x = train_matrix_x.reshape(train_matrix_x.shape[0]*train_matrix_x.shape[1],
                      train_matrix_x.shape[2], train_matrix_x.shape[3], train_matrix_x.shape[4])
test_matrix_x = test_matrix_x.reshape(test_matrix_x.shape[0]*test_matrix_x.shape[1],
                      test_matrix_x.shape[2], test_matrix_x.shape[3], test_matrix_x.shape[4])

#%%
#################
#CNN training
#################
grid_size = 25

random.seed(10)
np.random.seed(10)
tf.random.set_seed(10)

inputs_image = keras.Input(shape=(grid_size,grid_size,10)) 
x_img = layers.Conv2D(16, kernel_size=(3, 3), activation="relu")(inputs_image) 
x_img = layers.Conv2D(16, kernel_size=(3, 3), activation="relu")(x_img) 
x_img = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x_img) 
#x_img = layers.Conv2D(128, kernel_size=(3, 3), activation="relu")(x_img) 
x_img = layers.GlobalAveragePooling2D()(x_img)
x_img = layers.Dropout(0.5)(x_img) 
x_img = layers.Flatten()(x_img)

inputs_tab = keras.Input(shape=(9,))
x_tab = layers.Dense(16, activation="relu")(inputs_tab) 
x_tab = layers.Dropout(0.5)(x_tab)
x_tab = layers.Dense(16, activation="relu")(x_tab) 
x_tab = layers.Dropout(0.5)(x_tab) 
# x_tab = layers.Dense(16, activation="relu")(x_tab)
# x_tab = layers.Dropout(0.5)(x_tab) 
print("hi before")
print(x_tab.shape)
print(x_img.shape)

x = layers.concatenate([x_img, x_tab])
print(x.shape)
print("hi")

x = layers.Dense(256, activation="relu")(x) #out
x = layers.Dropout(0.5)(x) #out
# x = layers.Dense(256, activation="relu")(x) #64
# x = layers.Dropout(0.5)(x) 
# x = layers.Dense(256, activation="relu")(x) #32
# x = layers.Dropout(0.5)(x) #out
# x = layers.Dense(128, activation="relu")(x) #64
#x = layers.Dropout(0.5)(x) 
x = layers.Dense(128, activation="relu")(x) #32
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=[inputs_image, inputs_tab], outputs=outputs)
model.summary()

batch_size = 256 #128
epochs = 100 #100

optimizer = keras.optimizers.Adam(
	learning_rate=0.001, #5
	beta_1=0.9,
	beta_2=0.999,
	epsilon=1e-07,
	amsgrad=False,
	name="Adam"
)

    
auc =  keras.metrics.AUC(curve = "ROC", name = 'auc')
prauc = keras.metrics.AUC(curve = "PR", name = 'prauc')
model.compile(loss="binary_crossentropy", optimizer=optimizer,  #adam
              metrics=["accuracy", auc,
                       prauc])

# callback = keras.callbacks.EarlyStopping(
#     monitor="val_prauc",
#     min_delta=0,
#     patience=20,
#     restore_best_weights = True,
#     verbose = 1,
#     mode='max'
#     )

history = model.fit([train_matrix_x, train_tabular_x], train_y, batch_size=batch_size, epochs=epochs, verbose=2,
           validation_data=([test_matrix_x, test_tabular_x], test_y))

#%%

index_max_auc = np.argmax(history.history['val_auc'])
print("auc_max: index:", index_max_auc ,", auc:", history.history['val_auc'][index_max_auc],", prauc:", history.history['val_prauc'][index_max_auc])


index_max_prauc = np.argmax(history.history['val_prauc'])
print("prauc_max: index: ", index_max_prauc,", auc:", history.history['val_auc'][index_max_prauc],", prauc:", history.history['val_prauc'][index_max_prauc])



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('model accuracy')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['prauc'])
plt.plot(history.history['val_prauc'])
plt.title('model accuracy')
plt.ylabel('prauc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()




