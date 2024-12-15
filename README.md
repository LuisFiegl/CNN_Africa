# Conflict Forecasting in Africa using Remote Sensing Data
## About the Project
This repository contains all the code for the Statistical Consulting Project **Conflict Forecasting in Africa using Remote Sensing Data** at LMU Munich by Luis Fiegl and Ali Najibpour Nashi. 
[Associated report available here.](https://drive.google.com/drive/folders/1lDDNRQymG_GIfyfR11VdzBnzfWMKmDkM?usp=sharing)

### Abstract:
For many years, conflict prediction has been a fascinating subject. To develop early warning systems, researchers are employing statistical and mathematical methods to find conflict patterns in data. Particularly following the emergence of **spatial data** from satellites (remote sensing data) and various machine learning techniques suited to handle data of such complexity. In this project, we utilized such data to predict violent conflicts for the entire continent of Africa. We aimed to extend previous approaches by using **Convolutional Neural Networks (CNNs)** and **Sparse Convolutional Neural Networks (Sparce CNNs)**, to determine whether Deep Learning models can enhance prediction accuracy by utilizing spatial information.

## Data
*est_df.rds* represents the *tabular dataset*. It is available and doesnt have to be created. It can be used right away to fit the Random Forest (RF).

The *remote sensing feature matrix* mentioned in the report is not available, because it would take up too much storage space. The files in folder **data_preparation** can be used to create it. Still, the remote sensing data has to be downloaded in advance!

## Data Preparation
For replication, the paths to the data must be adjusted in every file. The files can be executed in the following order:
- *prepare_cell_polygons.R*: Creates all cells of Africa *africa_cell_polygons.rds*
- *prepare_eventdata.R*: Creates fatality data (matched to cells) *tabular_df.rds*
- *match_population_africa.R*: Matches cells to population data
- *match_rainfall_africa.R*: Matches cells to rainfall data
- *match_nighttimelights_africa.R*: Matches cells to nighttime lights data
- *landcover_data_download.ipynb*: Downloads the landcover data. It must be run for each year separately
- *vertical_landcover_match.R*: Matches cells to landcover data
- *landcover_year_merge.R*: Merges all years for the landcover data
- *match_Africa_data.R*: Matches all remote sensing features with the fatality data to create the *tabular dataset* *est_df.rds* and the *remote sensing feature matrix* *full_matrix.h5* which can be used to fit all of our models.

## Plots and Analysis
Contains files which we used to analyze our fatality data and create various plots (including map visualizations), used in the report.

## Prediction Models
This folder contains the models we used for our predictions. The RF in *random_forest.R* was implemented in R, using the *mlr3* framework. We implemented our CNN architecture in **Python** using the **Keras** library. Everything regarding the CNN is done in one file called *CNN.py*.

The architecture for the Sparse CNN is accessible in file *Sparse_CNN.py*, and its architecture is the same as the one we used for the standard CNN. The only thing that changed is that we introduce sparsity in the first convolutional layer. This is done with the function **SparseConv2D(filters=16, kernel size=(3,3), lam=0.0001).** The function introduces grouped sparsity as proposed in Kolb et al. (2024). Also note that for the Sparse CNN a **Keras version of 2.x** has to be used, since the function was only available for this Keras version for us. The code
of *Sparse_CNN.py* therefore differs a bit to the code for the standard CNN in *CNN.py*, which works on Keras **version 3.x.**. The file *SparseCNN_scaled_matrix.py* contains the Sparse CNN fitted on the transformed/scaled *remote sensing feature matrix*.

Additionally, we contain code to fit a standard DNN in *DNN.py*.
