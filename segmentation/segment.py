from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

os.chdir('.')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_segment_model(name, path):
    print('Initializing SAM model.....')
    sam = sam_model_registry[name](checkpoint=path)
    sam.to(device)
    predictor = SamPredictor(sam)
    print('Successfully Initialized')
    return predictor

def segment_image(predictor, user_select_image, x1, y1, x2, y2):
    predictor.set_image(np.array(user_select_image))
    mask, _, _ = predictor.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False)
    
    image_without_segment, mask_im = get_without_segment(mask, user_select_image)
    segmented_image = get_segment(mask, user_select_image)
    
    return image_without_segment, segmented_image, mask_im
    
def get_without_segment(mask, original_image):
    # Invert the mask
    mask_inv = 1 - mask
    mask_inv = np.stack([mask_inv]*3, axis=-1)

    # Apply the inverted mask to the original image
    image_without_segment = original_image * mask_inv
    image_without_segment[mask_inv == 0] = 255

    # Convert the result to a PIL Image
    image_without_segment_pil = Image.fromarray((image_without_segment[0]).astype(np.uint8))

    mask = mask[0]
    mask_inv = 1 - mask
    mask_inv = (mask_inv * 255).astype(np.uint8)
    kernel = np.ones((7,7),np.uint8)
    mask_inv_erode = cv2.erode(mask_inv, kernel, iterations = 5)
    mask_inv_erode_3c = np.stack([mask_inv_erode]*3, axis=-1)
    
    # Convert the inverted mask to a PIL Image
    mask_im = Image.fromarray((mask_inv_erode_3c).astype(np.uint8))
    
    return image_without_segment_pil, mask_im

def get_segment(mask, original_image):
    
    # Add an additional dimension to the mask to match the number of color channels in the image
    mask = np.expand_dims(mask, axis=-1)
    
    # Use the mask to get the part of the image corresponding to the highest probability class
    segmented_image = np.where(mask, original_image, 0)
    
    # Create an alpha channel where the segmented part is opaque and the rest is transparent
    alpha_channel = np.ones(mask.shape, dtype=np.uint8) * 255
    alpha_channel[mask == 0] = 0
    
    # Add the alpha channel to the segmented image
    segmented_image_rgba = np.concatenate([segmented_image, alpha_channel], axis=-1)

    segmented_image_rgba_pil = Image.fromarray((segmented_image_rgba[0]).astype(np.uint8))
    bbox = segmented_image_rgba_pil.getbbox()
    img_cropped = segmented_image_rgba_pil.crop(bbox)
    return img_cropped
