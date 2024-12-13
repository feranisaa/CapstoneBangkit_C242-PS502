# SIRA - Smart Interactive Reality for Adventure

## Team ID: CapstoneBangkit_C242-PS502
### Theme : Digital Experiences: Revolutionizing Sustainable Tourism
## Team Members:
### M312B4KX1617 – Ghina Puspamurti – Universitas Sebelas Maret - ML
### M312B4KY0012 – Abdee Ridho Pramono – Universitas Sebelas Maret - ML
### M312B4KX1713 – Hanifah Salsabila Ryadi – Universitas Sebelas Maret - ML
### C312B4KX0973 – Dana Ariska Susilowati – Universitas Sebelas Maret - CC
### C312B4KX1481 – Feranisa Kusuma Faustina – Universitas Sebelas Maret - CC
### A312B4KY1991 – Inzaghi Zhafran Aditya – Universitas Sebelas Maret - MD

## Application for helping tourism and local residents in Solo

SIRA aims to make it easier for tourists to find suitable destinations, support them to maximize their time, and introduce various tourist attractions and local cultures in Greater Solo that may not have been explored. SIRA app also focuses on helping to improve the local economy of Solo and its surrounding areas by equalizing tourism in Greater Solo. By providing personalized tourist destination recommendations, SIRA seeks to contribute to the growth of the tourism sector through an innovative technological approach. With the even distribution of tourists in Greater Solo, this can help improve the community's economy. 

## Machine Learning

Welcome to our Machine Learning Cohort project repository! This repository contains two machine learning models:

1. **Image Classification Model**
   - Built for classifying images using a custom dataset.
2. **Recommender System Model**
   - Provides recommendations based on preloaded online datasets.

Both models are implemented as Jupyter notebooks and designed to be run in Google Colab. Follow the instructions below to set up and use these models effectively.

---

## Prerequisites

Before proceeding, ensure you have the following:

1. **Google Account**
   - Required to use Google Colab and Google Drive.
2. **Basic Understanding of Jupyter Notebooks**
   - Familiarity with running cells in Colab.
3. **Internet Connection**
   - Necessary for downloading datasets and accessing Google Cloud Storage.

---

## Getting Started

### 1. Download the Notebooks

Download the notebooks from the following links:

- [Image Classification Notebook](https://github.com/feranisaa/CapstoneBangkit_C242-PS502/blob/main/ML/Image_Classification_FIX.ipynb)
- [Recommender System Notebook](https://github.com/feranisaa/CapstoneBangkit_C242-PS502/blob/main/RecommenderFIX.ipynb)

Save these files to a local directory or upload them directly to Google Colab.

---

## Using the Models

### **Image Classification Model**

1. **Download the Dataset**
   - [Click here to download the dataset](https://storage.googleapis.com/bucketsformodel/DatasetOke.zip).
   - Extract the dataset and upload it to your Google Drive.

2. **Set the Dataset Path**
   - Open the Image Classification notebook in Google Colab.
   - Locate the following lines of code:
     ```python
     # Check sub-directory
     !ls /content/drive/Shareddrives/Capstone/DatasetBaruLagi2
     DATA_DIR = "/content/drive/Shareddrives/Capstone/DatasetBaruLagi2"
     ```
   - Replace `DATA_DIR` with the path to your dataset in Google Drive. For example:
     ```python
     DATA_DIR = "/content/drive/MyDrive/MyDatasetFolder"
     ```

3. **Run the Notebook**
   - Execute all cells sequentially.

### **Recommender System Model**

1. **Dataset Preloaded**
   - The dataset for this model is hosted on Google Cloud Storage, so no manual dataset setup is required.

2. **Run the Notebook**
   - Open the Recommender System notebook in Google Colab.
   - Execute all cells sequentially.

---

## Notes

1. **Dependencies**
   - Both notebooks automatically install required libraries if missing.
   - Ensure you have the latest version of TensorFlow installed.

2. **Troubleshooting**
   - If you encounter issues with dataset paths, double-check that the path matches your Google Drive structure.
   - Ensure Colab is connected to the correct runtime (e.g., GPU for faster execution).

---

## Contact

For any questions or feedback, feel free to reach out to us via the repository's issue tracker.

