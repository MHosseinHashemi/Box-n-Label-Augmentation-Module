# BoXnLabelS

BoXnLabelS is a Python package for easy and customizable image augmentation, designed to generate augmented images and adjust their corresponding bounding box labels for deep learning models.

## Installation

Install via pip:

```bash
pip install BoXnLabelS
```

## Quick Start

```python
import BoXnLabelS as bls

# Initialize the augmentation object
augmentor = bls.Image_Custom_Augmentation(
    SP_intensity=0.2,  # Salt & Pepper noise intensity
    CWRO_Key=20,       # Clockwise rotation in degrees
    CCWRO_Key=20,      # Counterclockwise rotation in degrees
    Br_intensity=True, # Brightness adjustment
    H_Key=True,        # Horizontal flip
    V_Key=True,        # Vertical flip
    HE_Key=True,       # Histogram equalization
    GaussianBlur_KSize=5,  # Gaussian blur kernel size
    Random_Translation=True,  # Random translation
    Scaling_Range=(0.75, 1.25),  # Scaling range (min, max)
    Img_res=540  # Image resolution
)

# Apply augmentations to a dataset
augmentor.Generate_Data(input_path="input_directory", output_path="output_directory")
```

## Features

- **Noise Addition:** Salt & Pepper Noise
- **Image Enhancements:** Histogram Equalization, Brightness Adjustment
- **Transformations:** CW and CCW Random Rotations, H and V Flippings, Random Translation, Random Scaling
- **Blurring:** Gaussian Blur
- **Bounding Box Handling:** Automatic YOLO-format Bounding Box Augmentation

More to come ...

<br>

<img width="1010" alt="Vis" src="https://github.com/user-attachments/assets/f21f386b-0782-473c-b5ad-edb6c6555ccc" />


## Notes

- The module is under active development.
- Accepts images in **JPG format** only. (For now)
- Handles both labeled images (with bounding boxes) and unlabeled background images.
- Expect updates over a 4-month period, with regular improvements and enhancements.



# PyPl Page:
https://pypi.org/project/BoXnLabelS






