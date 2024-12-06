#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:43:59 2022

@author: ru79seh
"""
import structuredconvREAL

import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
from keras import layers
import keras

import pandas as pd 
from matplotlib import pyplot as plt
import random
import tensorflow as tf
import pickle
import os
import pyreadr
import h5py

#read in datasets
tabular_df = pyreadr.read_r('D:/consulting/code_consulting/Africa/data/est_df.rds')
tabular_df = tabular_df[None]

matrix_path = 'D:/consulting/code_consulting/Africa/data/full_matrix.h5'
with h5py.File(matrix_path, 'r') as f:
    full_matrix = np.array(f['full_matrix'])

full_matrix = np.transpose(full_matrix, (4, 3, 2, 1, 0))

#tabular dataset preperation
train_tabular = tabular_df[(tabular_df["year"]<=2018) & (tabular_df["year"]>2015)]
val_tabular = tabular_df[tabular_df["year"]==2019]
test_tabular = tabular_df[(tabular_df["year"]>2019)]

old_features = ["xcoord", "ycoord", "intersect_area", "ucdp_deaths_12_lag_1", "ucdp_deaths_12_lag_12", "ucdp_deaths_12_lag_24", "ucdp_deaths_12_neighbour_lag_1", "ucdp_deaths_12_neighbour_lag_12", "ucdp_deaths_12_neighbour_lag_24", "ucdp_12_conflict_since", "ucdp_12_neighbour_conflict_since"]

features_to_drop = ['row', 'col', 'year', 'month', 'ucdp_12_bin', 'ucdp_deaths_12', 'id', 'lag_pop', 'lag_rainfall', 'lag_nighttimes','lag_landcover_missing', 'lag_landcover_grass_shrub','lag_landcover_crop', 'lag_landcover_built', 'lag_landcover_water','lag_landcover_tree', 'lag_landcover_sea', 'lag_landcover_bare']
new_features = train_tabular.columns.difference(features_to_drop)

train_tabular_x = train_tabular[new_features].to_numpy()
train_y = train_tabular[["ucdp_12_bin"]].to_numpy()[:,0]

val_tabular_x = val_tabular[new_features].to_numpy()
val_y = val_tabular[["ucdp_12_bin"]].to_numpy()[:,0]

test_tabular_x = test_tabular[new_features].to_numpy()
test_y = test_tabular[["ucdp_12_bin"]].to_numpy()[:,0]



#################
#DNN training
#################
random.seed(33)
np.random.seed(33)
tf.random.set_seed(33)

inputs_tab = keras.Input(shape=(len(new_features),))
x_tab = layers.Dense(16)(inputs_tab)
x_tab = layers.BatchNormalization()(x_tab)
x_tab = layers.Activation("relu")(x_tab)
x_tab = layers.Dropout(0.5)(x_tab)

x_tab = layers.Dense(16)(x_tab)
x_tab = layers.BatchNormalization()(x_tab)
x_tab = layers.Activation("relu")(x_tab)
x_tab = layers.Dropout(0.5)(x_tab)
print(x_tab.shape)

x = layers.Dense(256)(x_tab) #out
#x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.5)(x) #out
# x = layers.Dense(256, activation="relu")(x) #64
# x = layers.Dropout(0.5)(x) 
# x = layers.Dense(256, activation="relu")(x) #32
# x = layers.Dropout(0.5)(x) #out
# x = layers.Dense(128, activation="relu")(x) #64
#x = layers.Dropout(0.5)(x) 
x = layers.Dense(128)(x) #32
#x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs_tab, outputs=outputs)
model.summary()

batch_size = 256 #128, 256 we tried putting it to 3000
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

history = model.fit(train_tabular_x, train_y, batch_size=batch_size, epochs=epochs, verbose=2,
           validation_data=(val_tabular_x, val_y))


index_max_auc = np.argmax(history.history['val_auc'])
print("auc_max: index:", index_max_auc ,", auc:", history.history['val_auc'][index_max_auc],", prauc:", history.history['val_prauc'][index_max_auc])


index_max_prauc = np.argmax(history.history['val_prauc'])
print("prauc_max: index: ", index_max_prauc,", auc:", history.history['val_auc'][index_max_prauc],", prauc:", history.history['val_prauc'][index_max_prauc])


# plots showcasing the training process
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


from sklearn.metrics import precision_recall_curve, auc
# Check how many we already predict as one:
predictions = model.predict(test_tabular_x)
print("Amount of predictions above 0.5: " + str(np.sum(predictions > 0.5)))

# Give the PRAUC
predicted_probabilities = predictions.flatten()
precision, recall, thresholds = precision_recall_curve(test_y, predicted_probabilities)
pr_auc = auc(recall, precision)
print(f'PR AUC: {pr_auc:.4f}')

# Find the optimal threshold
# Compute F1 score for each threshold
predictions = model.predict(train_tabular_x)
predicted_probabilities = predictions.flatten()
precision, recall, thresholds = precision_recall_curve(train_y, predicted_probabilities)

f1_scores = 2 * (precision * recall) / (precision + recall)
# Handle division by zero
f1_scores = np.nan_to_num(f1_scores)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

# Print the optimal threshold and corresponding F1 score
print(f'Optimal Threshold: {optimal_threshold:.4f}')
print(f'Train F1 Score at Optimal Threshold: {f1_scores[optimal_idx]:.4f}')

# use optimal threshold to predict test data
from sklearn.metrics import f1_score
predictions = model.predict(test_tabular_x)
predicted_probabilities = predictions.flatten()
predicted_classes = (predicted_probabilities >= optimal_threshold).astype(int)
f1 = f1_score(test_y, predicted_classes)
print(f'Test F1 Score: {f1:.4f}')

# Optimal threshold plot
f1_scores = f1_scores[:-1] 
plt.figure(figsize=(8, 6))
plt.plot(thresholds, f1_scores, marker='o', color='darkblue')
plt.xlabel('Threshold')
plt.ylabel('F1-Score')
#plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class_predictions = (predictions > optimal_threshold).astype(int)


cm = confusion_matrix(test_y, class_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()


######### SAVING PREDS FOR PLOTS
test_tabular["predictions"] = class_predictions
pyreadr.write_rds("D:/consulting/code_consulting/Africa/python_pred_results/19val_DNN_predictions.rds", test_tabular)