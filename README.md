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
My_data = Module.Image_Custom_Augmentation(
                                            SP_intensity=0.2,             # Salt and Pepper Intensity
                                            CWRO_Key=20,                  # CW Rotation Degree
                                            CCWRO_Key=20,                 # CCW Rotation Degree
                                            Br_intensity=True,            # Brightness Intensity
                                            H_Key = True,                 # Horizontal Flip
                                            V_Key = True,                 # Vertical Flip
                                            HE_Key= True,                 # Histogram Equalization
                                            GaussianBlur_KSize = 5,       # Gaussian Blur (Kernel Size, Kernel Size)
                                            Random_Translation = True,    # Random Translation (Shifting)
                                            Scaling_Range = (0.75, 1.25), # Random Scaling Range (Upscaling and Downscaling)
                                            Img_res=540                   # Image Resolution
)


#### Generate augmented data
My_data.Generate_Data(input_path="input_directory_path", output_path="output_directory_path")
```


<img width="1010" alt="Vis" src="https://github.com/user-attachments/assets/f21f386b-0782-473c-b5ad-edb6c6555ccc" />


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
