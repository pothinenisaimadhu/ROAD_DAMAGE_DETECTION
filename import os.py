import os
import torch
import cv2
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import torchvision
import numpy as np
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import retinanet_resnet50_fpn
from PIL import Image

import os
import cv2
import xml.etree.ElementTree as ET

def load_data(images_path, annotations_path):
    images = []
    annotations = []
    image_filenames_with_annotations = []

    # Step 1: Gather all image filenames that have corresponding annotations
    for xml_file in os.listdir(annotations_path):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(annotations_path, xml_file))
            root = tree.getroot()
            image_file = root.find('filename').text
            image_filenames_with_annotations.append(image_file)

    # Step 2: Load only the images that have corresponding annotations
    for image_file in os.listdir(images_path):
        if image_file in image_filenames_with_annotations:
            image_path = os.path.join(images_path, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)

    # Step 3: Load corresponding annotations
    for xml_file in os.listdir(annotations_path):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(annotations_path, xml_file))
            root = tree.getroot()
            image_file = root.find('filename').text
            if image_file in image_filenames_with_annotations:
                img_annotations = []
                for obj in root.findall('object'):
                    class_id = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    x_min = int(bndbox.find('xmin').text)
                    y_min = int(bndbox.find('ymin').text)
                    x_max = int(bndbox.find('xmax').text)
                    y_max = int(bndbox.find('ymax').text)
                    img_annotations.append((class_id, (x_min, y_min, x_max, y_max)))
                annotations.append((image_file, img_annotations))

    return images, annotations

# Paths to dataset
train_images_path = r"C:\Users\Sai Madhu\OneDrive\Desktop\code\road-damage-detection\rdd2020_pair\images"
train_annotations_path = r"C:\Users\Sai Madhu\OneDrive\Desktop\code\road-damage-detection\rdd2020_pair\annotations\xmls"

# Load data
train_images, train_annotations = load_data(train_images_path, train_annotations_path)

# Step 4: Print the number of images and annotations after filtering
print(f"Number of images: {len(train_images)}, Number of annotations: {len(train_annotations)}")








import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def visualize_random_samples(images, annotations, sample_size=6):
    """
    Visualizes random samples with bounding boxes in subplots.

    Parameters:
        images (list): List of images loaded using OpenCV.
        annotations (list): List of annotations corresponding to the images. 
                            Each annotation contains (image_file, [(class_id, (x_min, y_min, x_max, y_max))]).
        sample_size (int): Number of random samples to visualize.
    """
    # Select random samples
    indices = random.sample(range(len(images)), min(sample_size, len(images)))
    selected_images = [images[i] for i in indices]
    selected_annotations = [annotations[i] for i in indices]
    
    # Create subplots
    cols = 3  # Number of columns
    rows = (sample_size + cols - 1) // cols  # Calculate required rows
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten for easy iteration

    for i, (image, annotation) in enumerate(zip(selected_images, selected_annotations)):
        ax = axes[i]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for visualization
        ax.imshow(image)
        
        # Add bounding boxes
        for class_id, (x_min, y_min, x_max, y_max) in annotation[1]:
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x_min, y_min - 10, class_id, color='white', fontsize=10, backgroundcolor='red')

        # Set title and remove axes
        ax.set_title(annotation[0], fontsize=10)  # Display image filename
        ax.axis('off')

    # Remove any extra subplot axes
    for j in range(len(selected_images), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Visualize 5 random samples
visualize_random_samples(train_images, train_annotations, sample_size=6)









import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import xml.etree.ElementTree as ET
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import gc
import torch.cuda.amp as amp
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

class RoadDamageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.classes = ['background', 'D00', 'D10', 'D20', 'D40', 'D44']
        
        self.image_paths, self.annotation_paths = self._get_valid_pairs()
        print(f"Found {len(self.image_paths)} valid image-annotation pairs")
        
    def _get_valid_pairs(self):
        """Returns valid image-annotation pairs."""
        image_dir = os.path.join(self.root_dir, 'images')
        annot_dir = os.path.join(self.root_dir, 'annotations', 'xmls')
        
        valid_image_paths = []
        valid_annot_paths = []
        
        image_files = {f.lower(): f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        xml_files = {f.lower(): f for f in os.listdir(annot_dir) 
                    if f.lower().endswith('.xml')}
        
        for img_name_lower, img_name in image_files.items():
            xml_name = os.path.splitext(img_name_lower)[0] + '.xml'
            
            if xml_name in xml_files:
                img_path = os.path.join(image_dir, img_name)
                xml_path = os.path.join(annot_dir, xml_files[xml_name])
                
                try:
                    with Image.open(img_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                except:
                    continue
                
                boxes, labels = self._parse_annotation(xml_path)
                if len(boxes) > 0:
                    valid_image_paths.append(img_path)
                    valid_annot_paths.append(xml_path)
        
        return valid_image_paths, valid_annot_paths
    
    def _parse_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        img_width = float(root.find('size/width').text)
        img_height = float(root.find('size/height').text)
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text.split()[0]
            if class_name in self.classes:
                bbox = obj.find('bndbox')
                try:
                    xmin = max(0, float(bbox.find('xmin').text))
                    ymin = max(0, float(bbox.find('ymin').text))
                    xmax = min(img_width, float(bbox.find('xmax').text))
                    ymax = min(img_height, float(bbox.find('ymax').text))
                    
                    if xmin < xmax and ymin < ymax:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(self.classes.index(class_name))
                except (ValueError, AttributeError):
                    continue
        
        return boxes, labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        
        boxes, labels = self._parse_annotation(self.annotation_paths[idx])
        boxes = np.array(boxes)
        
        if self.transform:
            transformed = self.transform(image=img, bboxes=boxes, labels=labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        return img, target

def get_transform(train):
    if train:
        return A.Compose([
            A.RandomSizedBBoxSafeCrop(width=512, height=512, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def create_model(num_classes):
    model = models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Freeze some layers for transfer learning
    for name, parameter in model.backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)
    
    return model

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=50):
    model.to(device)
    
    # Learning rate scheduler with warmup
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2
    )
    
    scaler = amp.GradScaler()
    best_loss = float('inf')
    best_model_path = 'best_model.pth'
    patience = 10
    patience_counter = 0
    
    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            batch_count = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for images, targets in progress_bar:
                try:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    optimizer.zero_grad()
                    
                    with amp.autocast():
                        losses = model(images, targets)
                        loss = sum(loss for loss in losses.values())
                    
                    if not torch.isfinite(loss):
                        print('WARNING: non-finite loss, skipping batch')
                        continue
                    
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    
                    train_loss += loss.item()
                    batch_count += 1
                    
                    progress_bar.set_postfix({
                        'loss': loss.item(),
                        'lr': optimizer.param_groups[0]['lr']
                    })
                    
                    del images, targets, losses, loss
                    torch.cuda.empty_cache()
                
                except Exception as e:
                    print(f"Error in training batch: {str(e)}")
                    continue
            
            avg_train_loss = train_loss / batch_count
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for images, targets in val_loader:
                    try:
                        images = list(image.to(device) for image in images)
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                        
                        losses = model(images, targets)
                        loss = sum(loss for loss in losses.values())
                        val_loss += loss.item()
                        val_batch_count += 1
                        
                        del images, targets, losses, loss
                        torch.cuda.empty_cache()
                    
                    except Exception as e:
                        print(f"Error in validation batch: {str(e)}")
                        continue
            
            avg_val_loss = val_loss / val_batch_count
            
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {avg_val_loss:.4f}')
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, best_model_path)
                print(f'Saved best model with loss: {best_loss:.4f}')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
            
            gc.collect()
            torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Training interrupted: {str(e)}")
        # Save checkpoint even if interrupted
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, 'interrupted_model.pth')
    
    return model

def main():
    try:
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create datasets with augmentations
        train_transform = get_transform(train=True)
        val_transform = get_transform(train=False)
        
        dataset = RoadDamageDataset(
            root_dir="C:/Users/Sai Madhu/OneDrive/Desktop/code/road-damage-detection/rdd2020_pair",
            transform=train_transform,
            is_train=True
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        val_dataset.dataset.transform = val_transform
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,  # Reduced batch size
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x)),
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=2,  # Reduced batch size
            shuffle=False,
            collate_fn=lambda x: tuple(zip(*x)),
            num_workers=4,
            pin_memory=True
        )
        
        num_classes = len(dataset.classes)
        model = create_model(num_classes)
        
        # Initialize optimizer with weight decay
        optimizer = torch.optim.AdamW(
            [
                {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'backbone' not in n]},
                {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'backbone' in n],
                 'lr': 0.0001}
            ],
            lr=0.001,
            weight_decay=0.0001
        )
        
        # Train model
        trained_model = train_model(model, train_loader, val_loader, optimizer, device)
        
        # Save final model
        torch.save(trained_model.state_dict(), 'final_model.pth')
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()