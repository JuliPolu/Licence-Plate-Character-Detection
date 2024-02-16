import random
import os
import torch
from typing import List, Tuple
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import cv2
import matplotlib as plt


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



#Function to convert the XML with annotations to a pandas dataframe
def xml_to_dataframe(file_path: str) -> pd.DataFrame:
    # Parse the XML content
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract annotation data
    annotation_data = {
        "folder": root.findtext('folder'),
        "filename": root.findtext('filename'),
        "path": root.findtext('path'),
        "source": root.find('source/database').text,
        "width": int(root.find('size/width').text),
        "height": int(root.find('size/height').text),
        "depth": int(root.find('size/depth').text),
        "segmented": int(root.findtext('segmented')),
        "objects": []
    }
    
    # Loop through each object and get the details
    for obj in root.findall('object'):
        object_data = {
            "name": obj.findtext('name'),
            "pose": obj.findtext('pose'),
            "truncated": int(obj.findtext('truncated')),
            "difficult": int(obj.findtext('difficult')),
            "bndbox": {
                "xmin": int(obj.find('bndbox/xmin').text),
                "ymin": int(obj.find('bndbox/ymin').text),
                "xmax": int(obj.find('bndbox/xmax').text),
                "ymax": int(obj.find('bndbox/ymax').text)
            }
        }
        annotation_data["objects"].append(object_data)
    
    # Convert the extracted data to a pandas dataframe
    df = pd.json_normalize(annotation_data, record_path=['objects'], 
                           meta=['folder', 'filename', 'path', 'source', 'width', 'height', 'depth', 'segmented'])
    
    return df


#Function to visualize images with bboxes
def draw_bboxes(unique_images: List[str], datapath: str, folder: str, df: pd.DataFrame, figsize: Tuple[int, int] = (15, 5)):
    plt.figure(figsize=figsize)
    for i, image_file in enumerate(unique_images):
        img_path = os.path.join(datapath, folder, image_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Filter dataframe for the current image
        image_data = df[df['filename'] == image_file]

        # Plot each bounding box
        for index, row in image_data.iterrows():
            height, width, _ = image.shape
            x_min = int((row['center_X'] - row['width'] / 2) * width)
            y_min = int((row['center_y'] - row['height'] / 2) * height)
            x_max = int((row['center_X'] + row['width'] / 2) * width)
            y_max = int((row['center_y'] + row['height'] / 2) * height)

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
            cv2.putText(image, row['class_name'], (x_min, y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        # Create subplot for each image
        ax = plt.subplot(4, 6, i + 1)
        ax.imshow(image)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    
    
def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Calculate bounding box coordinates
def calculate_bbox(row):
    width, height = row['im_size']
    x_min = (row['center_X'] - row['width'] / 2) * width
    y_min = (row['center_y'] - row['height'] / 2) * height
    x_max = (row['center_X'] + row['width'] / 2) * width
    y_max = (row['center_y'] + row['height'] / 2) * height
    return x_min, y_min, x_max, y_max




