# Deep Vascular Topological Transformer Network (DVT-Net)

## Project Overview
DVT-Net is a deep learning model designed to analyze retinal images for the detection of vascular diseases. The model utilizes a Swin Transformer backbone integrated with advanced topological data analysis (TDA) techniques, specifically using persistence images derived from both Cubical Homology and Vietoris-Rips complexes. Additionally, the model employs an innovative patching technique to augment the dataset, enhancing the robustness and accuracy of the predictions.

## Model Architecture
The DVT-Net combines several cutting-edge technologies:
- **Swin Transformer**: Utilizes shifted window mechanisms for self-attention, adapting to the intrinsic spatial hierarchies of the image data.
- **Topological Data Analysis (TDA)**: Generates persistence images using both Cubical Homology and Vietoris-Rips complexes to capture topological features from the vascular structure in retinal images.
- **Image Patching**: Increases data diversity through systematic patching of original retinal images, improving model training and generalization.

The workflow is as follows:
1. **Pre-processing**: Retinal images are pre-processed and segmented to highlight vascular structures.
2. **Persistence Image Generation**: TDA techniques are applied to generate persistence images that summarize topological features.
3. **Patching**: The segmented images are divided into patches to create additional training data.
4. **Feature Extraction with Swin Transformer**: Extracts hierarchical features from both the persistence and patched images.
5. **Classification**: A fusion mechanism integrates features to classify the condition of the retina into categories such as Healthy or Diseased.

## Model Architecture
<img width="867" alt="Screenshot 2024-05-02 at 12 21 27 PM" src="https://github.com/SelimUCSD/BMENE4460-Final-Project/assets/63908462/4a205832-df58-494b-a3d7-e341e04436c1">
