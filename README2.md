<div id="top"></div>

<br />
<div align="center">

<h3 align="center">ECNN</h3>

  <p align="center">
    Brain Age Estimation based on Brain MRI by Ensemble of Deep Networks
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

Here is an implementation of ECNN model for the estimation brain age  based on MRI images. 
For more detail about the project please use [this link](https://ieeexplore.ieee.org/document/9377399) to access the paper.
[Pipeline]("/images/ECNN.png")

## Requirement installation

conda env create -f environment.yml

## Dataset
To train the proposed CNN networks, the Brain-Age Healthy Control (BAHC) dataset was used. This dataset contains 2001 Healthy individuals with a male/female ratio of 1016/985 with an average age of 18.12 ± 36.95 years. The age range of participants in this dataset is from 18 to 90 years. 

## Train the model

1. Run augement.py file for augment training data
2. Run ECNN_Train.py for train models

## Test the model

 Run ECNN_test.py for test models




