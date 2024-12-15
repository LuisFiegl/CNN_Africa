# Conflict Forecasting in Africa using Remote Sensing Data
## About the Project
This repository contains all the code for the Statistical Consulting Project **Conflict Forecasting in Africa using Remote Sensing Data** at LMU Munich by Luis Fiegl and Ali Najibpour Nashi. 
[Associated report available here.](https://drive.google.com/drive/folders/1lDDNRQymG_GIfyfR11VdzBnzfWMKmDkM?usp=sharing)

### Abstract:
For many years, conflict prediction has been a fascinating subject. To develop early warning systems, researchers are employing statistical and mathematical methods to find conflict patterns in data. Particularly following the emergence of **spatial data** from satellites (remote sensing data) and various machine learning techniques suited to handle data of such complexity. In this project, we utilized such data to predict violent conflicts for the entire continent of Africa. We aimed to extend previous approaches by using **Convolutional Neural Networks (CNNs)** and **Sparse Convolutional Neural Networks (Sparce CNNs)**, to determine whether Deep Learning models can enhance prediction accuracy by utilizing spatial information.

## About the Project
Due to our datasets (especially the Remote Sensing Feature Matrix) being too big, they are not contained in this repository. The files in folder **data_preparation** can be used to create all data which is used in the analysis.

We implemented our CNN architecture in **Python** using the **Keras** library. Everything
regarding the CNN is done in one file called CNN.py

The architecture for the Sparse CNN is accessible in file Sparse CNN.py, and its architecture is the same as the one we used for the standard CNN. The only thing that changed is that we introduce sparsity in the first convolutional layer. This is done with the function **SparseConv2D(filters=16, kernel size=(3,3),
lam=0.0001).** The function introduces grouped sparsity as proposed in Kolb et al. (2024). Also note that for the Sparse CNN a **Keras version of
2.x** has to be used, since the function is only available for this Keras version.

The code
of Sparse CNN.py therefore differs a bit to the code for the standard CNN in CNN.py,
which works on Keras **version 3.x.**
