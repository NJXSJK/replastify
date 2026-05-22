# 🚀 Replastify: Kaggle GPU Migration Guide

Since deep learning models (EfficientNet, ResNet) require significant compute power, training on a CPU is extremely slow. Kaggle provides **free GPUs** (like the Nvidia T4 x2 or P100) that can train these models in minutes instead of days.

This guide explains how to connect your GitHub repository to Kaggle, run the training pipeline, and download your trained models back to your PC.

---

## Step 1: Prepare Your Data Locally

Kaggle notebooks are ephemeral (they reset when you close them), so you need to upload your dataset to Kaggle permanently as a "Kaggle Dataset".

1.  On your local PC, navigate to your project folder: `/home/manas/Downloads/project/replastify/data/`
2.  Compress the `processed` folder into a zip file. You can do this in your terminal:
    ```bash
    cd /home/manas/Downloads/project/replastify/data/
    zip -r processed_data.zip processed/
    ```

## Step 2: Upload Data to Kaggle

1.  Go to [Kaggle.com](https://www.kaggle.com/) and log in (or create an account).
2.  Click the **"+" (Create)** button in the left sidebar and select **Dataset**.
3.  Name your dataset: `replastify-data`
4.  Drag and drop the `processed_data.zip` file you just created.
5.  Click **Create**. Wait for the upload to finish.

## Step 3: Set Up the Kaggle Notebook

1.  Click the **"+" (Create)** button again and select **Notebook**.
2.  **Turn on the GPU:**
    *   Look at the right-hand sidebar.
    *   Under **Notebook Options**, find **Accelerator**.
    *   Change it from "None" to **GPU T4 x2** (or GPU P100).
    *   *Note: Kaggle will ask you to verify your phone number to use GPUs if you haven't already.*
3.  **Attach your Dataset:**
    *   In the right-hand sidebar, click **Add Input** (or "Add Data").
    *   Select **Your Datasets** and add the `replastify-data` dataset you uploaded in Step 2.

---

## Step 4: Run the Code

Now that your environment is ready, you will run a series of commands in the notebook cells to pull your code from GitHub and execute it.

Copy and paste each of these blocks into a new cell in your Kaggle notebook and run them in order.

### Cell 1: Clone Your GitHub Repository
We will pull the latest code from your GitHub repo.
```python
!git clone https://github.com/NJXSJK/replastify.git
%cd replastify
```

### Cell 2: Install Dependencies
```python
!pip install -r backend/requirements.txt
```

### Cell 3: Fix the File Paths for Kaggle
Your Python scripts currently point to local folders on your PC. We need to point them to Kaggle's `/kaggle/input/` (where your dataset is) and `/kaggle/working/` (where we can save files). 

Run this cell to automatically update the paths in both scripts:
```python
import sys
from pathlib import Path

def update_paths(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the Path objects with string paths for Kaggle
    content = content.replace(
        'DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"',
        'DATA_DIR = Path("/kaggle/input/replastify-data/processed")'
    )
    content = content.replace(
        'OUTPUT_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "outputs"',
        'OUTPUT_DIR = Path("/kaggle/working/outputs")'
    )
    content = content.replace(
        'MODEL_DIR = Path(__file__).resolve().parent.parent / "backend" / "models"',
        'MODEL_DIR = Path("/kaggle/working/models")'
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
        
update_paths("notebooks/03_train_model.py")
update_paths("notebooks/04_evaluate_model.py")
print("✅ Paths updated successfully for Kaggle!")
```

### Cell 4: Train the Models
This will take a few hours depending on the GPU. It will train both EfficientNet-B0 and ResNet50.
```python
!python notebooks/03_train_model.py
```

### Cell 5: Evaluate the Models
This will generate your confusion matrix, F1-scores, and Grad-CAM visualizations.
```python
!python notebooks/04_evaluate_model.py
```

---

## Step 5: Download Your Trained Models

Once training and evaluation are complete, your trained models (`.pth` files) and visualization images (`.png` files) will be saved in Kaggle's working directory. 

To easily download them back to your PC, run this final cell to zip them up:

### Cell 6: Zip Outputs for Download
```python
import shutil

# Zip the models folder
shutil.make_archive('/kaggle/working/trained_models', 'zip', '/kaggle/working/models')

# Zip the outputs (graphs, matrices) folder
shutil.make_archive('/kaggle/working/evaluation_outputs', 'zip', '/kaggle/working/outputs')

print("✅ Zipped successfully! Look in the right-hand panel under 'Output' to download them.")
```

**To Download:**
1. Look at the right-hand sidebar in Kaggle.
2. Scroll down to the **Output** section.
3. You will see `trained_models.zip` and `evaluation_outputs.zip`.
4. Hover over them, click the three dots (`...`), and select **Download**.

---

## Final Step (Local PC)

1. Extract the downloaded `trained_models.zip`.
2. Move the `best_efficientnet_b0.pth` (or whichever won) into your local `/home/manas/Downloads/project/replastify/backend/models/` folder.
3. You are now ready to start Phase 3 (Backend API) using your newly trained, highly accurate model!
