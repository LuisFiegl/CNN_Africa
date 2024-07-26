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
#tabular_df = pd.read_csv('D:/consulting/code_consulting/Example Code/Example Code/data/cnn_est_df.csv')

#%%
#tabular prep
train_tabular = tabular_df[(tabular_df["year"]<=2019) & (tabular_df["year"]>2015)]
test_tabular = tabular_df[(tabular_df["year"]>2019)]

old_features = ["xcoord", "ycoord", "intersect_area", "ucdp_deaths_12_lag_1", "ucdp_deaths_12_lag_12", "ucdp_deaths_12_lag_24", "ucdp_deaths_12_neighbour_lag_1", "ucdp_deaths_12_neighbour_lag_12", "ucdp_deaths_12_neighbour_lag_24", "ucdp_12_conflict_since", "ucdp_12_neighbour_conflict_since"]
features_to_drop = ['year', 'month', 'ucdp_12_bin', 'ucdp_deaths_12', 'id']

new_features = train_tabular.columns.difference(features_to_drop)

train_tabular_x = train_tabular[old_features].to_numpy()
train_y = train_tabular[["ucdp_12_bin"]].to_numpy()[:,0]


test_tabular_x = test_tabular[old_features].to_numpy()

test_y = test_tabular[["ucdp_12_bin"]].to_numpy()[:,0]

#%%

#matrix prep
train_matrix_x = full_matrix[0:4]
print(train_matrix_x.shape)
test_matrix_x = full_matrix[4:5]

#train_matrix_x = np.repeat(train_matrix_x[:, np.newaxis, :, :, :], train_tabular_x.shape[0]/full_matrix.shape[1], axis=1)
train_matrix_x = np.repeat(train_matrix_x[:, np.newaxis, :, :, :], 12, axis=1)
test_matrix_x = np.repeat(test_matrix_x[:, np.newaxis, :, :, :], test_tabular_x.shape[0]/full_matrix.shape[1], axis=1)
print(test_matrix_x.shape)

#%%
train_matrix_x = train_matrix_x.reshape(train_matrix_x.shape[0]*train_matrix_x.shape[1]*train_matrix_x.shape[2],
                      train_matrix_x.shape[3], train_matrix_x.shape[4], train_matrix_x.shape[5])
test_matrix_x = test_matrix_x.reshape(test_matrix_x.shape[0]*test_matrix_x.shape[1]*test_matrix_x.shape[2],
                      test_matrix_x.shape[3], test_matrix_x.shape[4], test_matrix_x.shape[5])

#%%
#################
#CNN training
#################
random.seed(12)
np.random.seed(12)
tf.random.set_seed(12)

inputs_tab = keras.Input(shape=(11,))
x = layers.Dense(16, activation="relu")(inputs_tab) 
x = layers.Dropout(0.5)(x)
x = layers.Dense(16, activation="relu")(x) 
x = layers.Dropout(0.5)(x) 
# x_tab = layers.Dense(16, activation="relu")(x_tab)
# x_tab = layers.Dropout(0.5)(x_tab) 
print(x.shape)

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
model = keras.Model(inputs=inputs_tab, outputs=outputs)
model.summary()

batch_size = 256 #128 256
epochs = 300 #100

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

history = model.fit(train_tabular_x, train_y, batch_size=batch_size, epochs=epochs, verbose=2,
           validation_data=(test_tabular_x, test_y))

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
plt.title('model auc')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['prauc'])
plt.plot(history.history['val_prauc'])
plt.title('model prauc')
plt.ylabel('prauc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()





# %%


