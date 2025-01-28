# Lung X-Ray Anomaly Detection with Explainable AI

## Overview

This project detects pneumonia and other lung anomalies in chest X-ray images using deep learning techniques and Explainable AI (XAI). The approach focuses on accurate diagnosis while highlighting the regions influencing the model's decisions. The implementation is available as a single Google Colab notebook for ease of use and reproducibility.

---

## Features

1. **Pneumonia Detection**:
   - Uses a VGG16-based deep learning model for classification.
   - Focuses on lung-segmented areas for better precision.The segmentation is done using U-NET diffusion model.

2. **Explainable AI (XAI)**:
   - **Grad-CAM**: Highlights critical regions in X-ray images influencing the model's decisions.
   - **Modified LIME**: Generates explanations by applying a customized LIME approach in the VAE latent space, using K-Nearest Neighbors (KNN) for selecting similar cases.

3. **Medical Reporting**:
   - Automatically generates professional medical reports with annotated X-ray images, metrics, and observations.

---


## Datasets

- **Chest X-Ray Dataset**: [Paul Mooney's dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **Lung Segmentation Dataset**: [Nitsan Ben Hanoch's dataset](https://www.kaggle.com/nitsanbenhanoch/pre-processed-xray-lungs-segmentation)

---

## Requirements

This notebook uses the Google Colab environment, which comes pre-installed with most required libraries. However, if running locally, install the dependencies with:

```bash
pip install tensorflow scikit-learn matplotlib Pillow
