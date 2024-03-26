# Box n Label Augmentation Module
This Python package provides a customizable image augmentation toolset primarily aimed at generating augmented images and their corresponding labels for training deep learning models. It offers various augmentation techniques such as Salt and Pepper Noise, Brightness adjustment, Horizontal and Vertical Flipping, Histogram Equalization, and Rotation.

## Installation
Clone this repository to your local directory:
```bash
git clone https://github.com/MHosseinHashemi/Box-n-Label-Augmentation-Module.git
```

## Usage

```python
from Image_Custom_Augmentation import Image_Custom_Augmentation

#### Initialize Image Custom Augmentation object
my_data = Image_Custom_Augmentation(SP_intensity=False,  # Salt and Pepper Intensity
                                    RO_Key=False,        # Rotation Intensity
                                    Br_intensity=False,  # Brightness Intensity
                                    H_Key=False,         # Horizontal Flip
                                    V_Key=False,         # Vertical Flip
                                    HE_Key=False,        # Histogram Equalization
                                    Img_res= 540)        # Image Resolution

#### Generate augmented data
my_data.Generate_Data(input_path="input_directory_path", output_path="output_directory_path")
```

## Parameters

- **SP_intensity**: A float value indicating the intensity of Salt and Pepper Effect. Higher values mean more salt and pepper noise added to the images.
- **RO_Key**: An integer value indicating the rotation intensity. Positive values represent clockwise rotation.
- **Br_intensity**: An integer value indicating the brightness intensity. Higher values mean brighter images (for positive values) or darker images (for negative values).
- **H_Key**: A boolean value indicating whether to generate horizontally flipped samples.
- **V_Key**: A boolean value indicating whether to generate vertically flipped samples.
- **HE_Key**: A boolean value indicating whether to generate histogram equalized samples.
- **Img_res**: An integer value indicating the image resolution. Default value is 540. (540*540)

## Usage Examples
<img width="1250" alt="Capture" src="https://github.com/MHosseinHashemi/Box-n-Label-Augmentation-Module/assets/90381570/bde7df2a-8b4e-47fe-bd9c-e664d3959dfc">



## Notes

- **This module is currently under developement!**
- **The module initially designed to be a tool in data preprocessing for binary classification task, however we seek to enable it to work for multiclass cases**
- Input images must be in JPG format.
- The tool generates augmented labels for images with corresponding bounding box labels in YOLO format.
- The module exclusively handles augmentation for non-target samples, i.e., images without labels (also known as background samples), and generates augmented images accordingly.
- If the matching label file for each image sample is not present in the folder, the module treats them as background samples.

  
## Dependencies

- OpenCV (`cv2`)
- NumPy (`numpy`)
- tqdm (`tqdm`)
- Matplotlib (`matplotlib`)
