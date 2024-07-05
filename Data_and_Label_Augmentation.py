import os
import cv2
import shutil
import math
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Image_Custom_Augmentation:

    def __init__(self, 
                    SP_intensity = False,
                    CWRO_Key = False,
                    CCWRO_Key = False,
                    Br_intensity = False,
                    H_Key = False,
                    V_Key = False,
                    HE_Key = False,
                    GaussianBlurr_KSize = False,
                    Random_Translation = False,
                    Img_res = 540):
        
        # Salt and Pepper Intensity key
        self.SP_intensity = SP_intensity 
        # Brightness Intensity key
        self.Br_intensity = Br_intensity 
        # Horizontal Flip Key
        self.H_Key = H_Key 
        # Vertical Flip Key
        self.V_Key = V_Key 
        # CW Rotate Key
        self.CWRO_Key = CWRO_Key 
        # CCW Rotate Key
        self.CCWRO_Key = CCWRO_Key
        # Histogram Equalization Key
        self.HE_Key = HE_Key 
        # Gaussian Blurring key
        self.GaussianBlurr_KSize = GaussianBlurr_KSize
        # Random Translation key
        self.Random_Translation = Random_Translation
        # Image Resolution key
        self.Img_res = Img_res
        
        
    def Salt_n_Pepper(self, image_path, output_dir):
        image = cv2.imread(image_path)
        clean_label = os.path.splitext(os.path.basename(image_path))[0]
        # Create a Salt&Pepper filter
        height, width, _ = image.shape

        # Generate random noise mask
        salt_mask = np.random.rand(height, width) < self.SP_intensity
        pepper_mask = np.random.rand(height, width) < self.SP_intensity
        
        # Apply salt noise
        image[salt_mask] = [255,255,255]  # Set pixel to white (salt)
        image[pepper_mask] = [0,0,0]  # Set pixel to black (pepper)

        # Save the modified image to the output path
        custom_name = f"{clean_label}"+"_SP_"+".jpg"
        output_path = os.path.join(output_dir, custom_name)
        cv2.imwrite(output_path, image)
        
        # Reset
        del salt_mask, pepper_mask, image, clean_label, output_path, custom_name
            
            
    def Histogram_Equalization(self, image_path, output_dir):
        image = cv2.imread(image_path)
        clean_label = os.path.splitext(os.path.basename(image_path))[0]

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(gray_image)

        # Save the modified image to the output path
        custom_name = f"{clean_label}"+"_HE_"+".jpg"
        output_path = os.path.join(output_dir, custom_name)
        cv2.imwrite(output_path, equalized_image)

        # Reset
        del equalized_image, gray_image, image, clean_label, custom_name, output_path
     
    
    def Rotate(self, image_path, output_dir):
        image = cv2.imread(image_path)
        clean_label = os.path.splitext(os.path.basename(image_path))[0]
        
        # Apply Rotation
        height, width = image.shape[:2]
        centerX, centerY = (width // 2, height // 2)
        
        "Get the rotation Matrix and apply it to the image"
        M_1 = cv2.getRotationMatrix2D((centerX, centerY), -self.CWRO_Key, 1.0) # Clock Wise Rotation Matrix
        M_2 = cv2.getRotationMatrix2D((centerX, centerY), self.CCWRO_Key, 1.0) # C-Clock Wise Rotation Matrix
        cw_rotated_image = cv2.warpAffine(image, M_1, (width, height))
        ccw_rotated_image = cv2.warpAffine(image, M_2, (width, height))
    
        # Save the modified image to the output path
        custom_name_1 = f"{clean_label}"+"_CWRO_"+".jpg"
        custom_name_2 = f"{clean_label}"+"_CCWRO_"+".jpg"
        output_path_1 = os.path.join(output_dir, custom_name_1)
        output_path_2 = os.path.join(output_dir, custom_name_2)
        cv2.imwrite(output_path_1, cw_rotated_image)
        cv2.imwrite(output_path_2, ccw_rotated_image)
        
        # Reset
        del cw_rotated_image,ccw_rotated_image, custom_name_1,custom_name_2, output_path_1,output_path_2, image, clean_label
    
    
    def Brightness(self, image_path, output_dir):
        image = cv2.imread(image_path)
        clean_label = os.path.splitext(os.path.basename(image_path))[0]
        # Create a 2D plate of same values to Add/Subtract from the initial image
        plate = np.ones(image.shape, dtype="uint8") * (self.Br_intensity)
        # Two different filters (Br/Da)
        brighter_img = cv2.add(image, plate)
        darker_img = cv2.subtract(image, plate)
        
        # Save the modified images to the output path
        custom_name_1 = f"{clean_label}"+"_BR_"+".jpg"
        custom_name_2 = f"{clean_label}"+"_DA_"+".jpg"
        output_path_1 = os.path.join(output_dir, custom_name_1)
        output_path_2 = os.path.join(output_dir, custom_name_2)
        cv2.imwrite(output_path_1, brighter_img)
        cv2.imwrite(output_path_2, darker_img)
        
        # Reset
        del brighter_img, darker_img, image, clean_label, custom_name_1, custom_name_2, output_path_1, output_path_2
        
        
    
    def Flip_V(self, image_path, output_dir):
        image = cv2.imread(image_path)
        clean_label = os.path.splitext(os.path.basename(image_path))[0]
        # Flip the image
        V_flipped = cv2.flip(image, 0)
        # Save the modified images to the output path
        custom_name_1 = f"{clean_label}"+"_VF_"+".jpg"
        output_path_1 = os.path.join(output_dir, custom_name_1)
        cv2.imwrite(output_path_1, V_flipped)
        
        # Reset
        del V_flipped, image, clean_label, custom_name_1, output_path_1
    
    
    
    def Flip_H(self, image_path, output_dir):
        image = cv2.imread(image_path)
        clean_label = os.path.splitext(os.path.basename(image_path))[0]
        # Flip the image
        H_flipped = cv2.flip(image, 1)
        # Save the modified images to the output path
        custom_name_1 = f"{clean_label}"+"_HF_"+".jpg"
        output_path_1 = os.path.join(output_dir, custom_name_1)
        cv2.imwrite(output_path_1, H_flipped)
        
        # Reset
        del H_flipped, image, clean_label, custom_name_1, output_path_1  



    def GaussianBlurr(self ,image_path, output_dir):
        image = cv2.imread(image_path)
        clean_label = os.path.splitext(os.path.basename(image_path))[0]
        # Apply the Gaussian Blurring Filter  
        GBlurred = cv2.GaussianBlur(image, (self.GaussianBlurr_KSize, self.GaussianBlurr_KSize), 0)
        # Save the modified images to the output path
        custom_name_1 = f"{clean_label}"+"_GBlurr_"+".jpg"
        output_path_1 = os.path.join(output_dir, custom_name_1)
        cv2.imwrite(output_path_1, GBlurred)

        # Reset
        del GBlurred, image, clean_label, custom_name_1, output_path_1
      

    """Helper function"""
    # generates random affine transformation matrix
    def Transformation_Matrix_Generator(self, w, h):
        t_value = int((h + w)/10) 
        matrix = np.float32([
            [1, 0, random.randint(-t_value, t_value)],
            [0, 1, random.randint(-t_value, t_value)]
        ])

        return matrix



    def Image_translation(self, image_path, output_dir):
        image = cv2.imread(image_path)
        clean_label = os.path.splitext(os.path.basename(image_path))[0]
        height, width = image.shape[:2] 
        # call the helper function
        M = self.Transformation_Matrix_Generator(width, height)
        # Apply Random Shifting  
        Shifted = cv2.warpAffine(image, M, (width, height))
        # Save the modified images to the output path
        custom_name_1 = f"{clean_label}"+"_Shifted_"+".jpg"
        output_path_1 = os.path.join(output_dir, custom_name_1)
        cv2.imwrite(output_path_1, Shifted)

        # Reset
        del Shifted, image, clean_label, custom_name_1, output_path_1, height, width

        return M



    @staticmethod
    def translation_mapper(x, y, matrix, width, height):
        # Normalize
        Xi = int(x * width)
        Yi = int(y * height)
        # translation coordinates
        ho_shift = matrix[0][2]
        ve_shift = matrix[1][2]
        # shift the coordinates
        Xj = Xi + ho_shift
        Yj = Yi + ve_shift

        return Xj, Yj


    @staticmethod    
    def rotation_mapper(img_size, alpha, Xi, Yi):
        #"""Step 1"""
        Xi = int(Xi * img_size - (img_size//2))
        Yi = int(Yi * img_size - (img_size//2))

        #"Step 2"
        alpha = math.radians(alpha)
        Xj = int(Xi*math.cos(alpha) - Yi*math.sin(alpha))
        Yj = int(Xi*math.sin(alpha) + Yi*math.cos(alpha))
        Xj += (img_size//2)
        Yj += (img_size//2)
        return Xj, Yj
    
    
    @staticmethod
    def h_flip_mapper(img_size, horizontal_key, Xi, Yi):
        if horizontal_key == True:
            Xj = (img_size) - ((img_size) * Xi)
            Yj = (img_size) * Yi
        # else:
        #     print("Error!: 1) Nothing to flip .... or ... 2) Flags are not correct ...")
            
        return int(Xj), int(Yj)


    @staticmethod
    def v_flip_mapper(img_size, vertical_key, Xi, Yi):
        if vertical_key == True:
            Xj = img_size * Xi
            Yj = img_size - (img_size * Yi)
            
        return int(Xj), int(Yj)
    
        
    def Generate_Data(self, input_path, output_path):
        for index in tqdm(os.listdir(input_path)):
            if ".jpg" in index:
                image_path = os.path.join(input_path, index)

                "New path defined for label file"
                label_path = os.path.join(input_path, index.rstrip(".jpg")+".txt")

                # Switching between functions
                if self.GaussianBlurr_KSize:
                    self.GaussianBlurr(image_path, output_dir=output_path)
                    """Bounding Box Augmentation"""
                    clean_label = os.path.splitext(os.path.basename(label_path))[0]
                    if "T" in clean_label:
                        custom_name_1 = f"{clean_label}"+"_GBlurr_"+".txt"
                        output_path_1 = os.path.join(output_path, custom_name_1)
                        shutil.copyfile(label_path, output_path_1)


                if self.Random_Translation:
                    mat = self.Image_translation(image_path, output_dir=output_path)
                    """Bounding Box Augmentation"""
                    clean_label = os.path.splitext(os.path.basename(label_path))[0]
                    if "T" in clean_label:
                        custom_name = f"{clean_label}"+"_Shifted_"+".txt"
                        output_label_path = os.path.join(output_path, custom_name)

                        # 1st: Open the Original Label Text File and read All values from it
                        with open(label_path, "r") as input_file:
                            with open(output_label_path, "w") as output_file:
                                for line in input_file:
                                    temp_list = [float(word) for word in line.split()]
                                    x,y = temp_list[1:3]
                                    # 2nd: Call the translation mapper function to rotate the X,Y values
                                    x,y = self.translation_mapper(x, y, matrix=mat, width=self.Img_res, height=self.Img_res)
                                    # 3rd: Revert them back to the original YOLO format by and divide them by 540
                                    x /=self.Img_res
                                    y /=self.Img_res
                                    temp_list[1:3] = x,y
                                    temp_list[0] = int(temp_list[0])
                                    # 4th: Save the new values to the Label File and put it inside a suitable dir
                                    output_file.write(' '.join(map(str, temp_list)) + '\n')


                if self.H_Key:
                    self.Flip_H(image_path, output_dir=output_path)
                    """Bounding Box Augmentation"""
                    clean_label = os.path.splitext(os.path.basename(label_path))[0]
                    if "T" in clean_label:
                        custom_name = f"{clean_label}"+"_HF_"+".txt"
                        output_label_path = os.path.join(output_path, custom_name)

                        # 1st: Open the Original Label Text File and read All values from it
                        with open(label_path, "r") as input_file:
                            with open(output_label_path, "w") as output_file:
                                for line in input_file:
                                    temp_list = [float(word) for word in line.split()]
                                    x,y = temp_list[1:3]
                                    # 2nd: Call the rotation mapper function to rotate the X,Y values
                                    x,y = self.h_flip_mapper(self.Img_res, self.H_Key, x, y)
                                    # 3rd: Revert them back to the original YOLO format by and divide them by 540
                                    x /=self.Img_res
                                    y /=self.Img_res
                                    temp_list[1:3] = x,y
                                    temp_list[0] = int(temp_list[0])
                                    # 4th: Save the new values to the Label File and put it inside a suitable dir
                                    output_file.write(' '.join(map(str, temp_list)) + '\n')
                        
                
                
                if self.V_Key:
                    self.Flip_V(image_path, output_dir=output_path)
                    """Bounding Box Augmentation"""
                    clean_label = os.path.splitext(os.path.basename(label_path))[0]
                    if "T" in clean_label:
                        custom_name = f"{clean_label}"+"_VF_"+".txt"
                        output_label_path = os.path.join(output_path, custom_name)

                        # 1st: Open the Original Label Text File and read All values from it
                        with open(label_path, "r") as input_file:
                            with open(output_label_path, "w") as output_file:
                                for line in input_file:
                                    temp_list = [float(word) for word in line.split()]
                                    x,y = temp_list[1:3]
                                    # 2nd: Call the rotation mapper function to rotate the X,Y values
                                    x,y = self.v_flip_mapper(self.Img_res, self.V_Key, x, y)
                                    # 3rd: Revert them back to the original YOLO format by and divide them by 540
                                    x /=self.Img_res
                                    y /=self.Img_res
                                    temp_list[1:3] = x,y
                                    temp_list[0] = int(temp_list[0])
                                    # 4th: Save the new values to the Label File and put it inside a suitable dir
                                    output_file.write(' '.join(map(str, temp_list)) + '\n')
             
                    

                if self.Br_intensity:
                    self.Brightness(image_path, output_dir=output_path)
                    """Bounding Box Augmentation"""
                    clean_label = os.path.splitext(os.path.basename(label_path))[0]
                    if "T" in clean_label:
                        custom_name_1 = f"{clean_label}"+"_BR_"+".txt"
                        custom_name_2 = f"{clean_label}"+"_DA_"+".txt"
                        output_path_1 = os.path.join(output_path, custom_name_1)
                        output_path_2 = os.path.join(output_path, custom_name_2)
                        shutil.copyfile(label_path, output_path_1)
                        shutil.copyfile(label_path, output_path_2)


                if self.HE_Key:
                    self.Histogram_Equalization(image_path, output_dir=output_path)
                    """Bounding Box Augmentation"""
                    clean_label = os.path.splitext(os.path.basename(label_path))[0]
                    if "T" in clean_label:
                        custom_name = f"{clean_label}"+"_HE_"+".txt"
                        output_label_path = os.path.join(output_path, custom_name)
                        shutil.copyfile(label_path, output_label_path)

                    
                if self.CWRO_Key:
                    self.Rotate(image_path, output_dir=output_path)
                    """Bounding Box Augmentation"""
                    clean_label = os.path.splitext(os.path.basename(label_path))[0]
                    if "T" in clean_label:  
                        custom_name = f"{clean_label}"+"_CWRO_"+".txt"
                        output_label_path = os.path.join(output_path, custom_name)
                    
                        # 1st: Open the Original Label Text File and read All values from it
                        with open(label_path, "r") as input_file:
                            with open(output_label_path, "w") as output_file:
                                for line in input_file:
                                    temp_list = [float(word) for word in line.split()]
                                    x,y = temp_list[1:3]
                                    # 2nd: Call the rotation mapper function to rotate the X,Y values
                                    x,y = self.rotation_mapper(self.Img_res, self.CWRO_Key, x, y)
                                    # 3rd: Revert them back to the original YOLO format by and divide them by Resolution
                                    x /=self.Img_res
                                    y /=self.Img_res
                                    temp_list[1:3] = x,y
                                    temp_list[0] = int(temp_list[0])
                                    # 4th: Save the new values to the Label File and put it inside a suitable dir
                                    output_file.write(' '.join(map(str, temp_list)) + '\n')
                
                if self.CCWRO_Key:
                    self.Rotate(image_path, output_dir=output_path)
                    """Bounding Box Augmentation"""
                    clean_label = os.path.splitext(os.path.basename(label_path))[0]
                    if "T" in clean_label:  
                        custom_name = f"{clean_label}"+"_CCWRO_"+".txt"
                        output_label_path = os.path.join(output_path, custom_name)
                    
                        # 1st: Open the Original Label Text File and read All values from it
                        with open(label_path, "r") as input_file:
                            with open(output_label_path, "w") as output_file:
                                for line in input_file:
                                    temp_list = [float(word) for word in line.split()]
                                    x,y = temp_list[1:3]
                                    # 2nd: Call the rotation mapper function to rotate the X,Y values
                                    """Why 30?"""
                                    # x,y = self.rotation_mapper(self.Img_res,30,x,y)
                                    x,y = self.rotation_mapper(self.Img_res, -self.CCWRO_Key, x, y)
                                    # 3rd: Revert them back to the original YOLO format by and divide them by Resolution
                                    x /=self.Img_res
                                    y /=self.Img_res
                                    temp_list[1:3] = x,y
                                    temp_list[0] = int(temp_list[0])
                                    # 4th: Save the new values to the Label File and put it inside a suitable dir
                                    output_file.write(' '.join(map(str, temp_list)) + '\n')
                    
                    

                if self.SP_intensity:
                    self.Salt_n_Pepper(image_path, output_dir=output_path)
                    """Bounding Box Augmentation"""
                    clean_label = os.path.splitext(os.path.basename(label_path))[0]
                    if "T" in clean_label:
                        custom_name = f"{clean_label}"+"_SP_"+".txt"
                        output_label_path = os.path.join(output_path, custom_name)
                        shutil.copyfile(label_path, output_label_path)


            elif (".jpg" not in index) and (".txt" not in index) :
                print("Error! No functionality has been called or perhaps your files are not in .jpg format")
            


# Good Luck
# MH
