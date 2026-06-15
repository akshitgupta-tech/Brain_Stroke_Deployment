"""
Preprocessing pipeline with CLAHE, alignment, and normalization
"""
import cv2
import numpy as np
import dlib
from PIL import Image
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FacePreprocessor:
    def __init__(self, img_size=224, use_clahe=True):
        self.img_size = img_size
        self.use_clahe = use_clahe
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # dlib face detector and predictor paths
        self.detector = dlib.get_frontal_face_detector()
        
    def detect_face(self, image):
        """Detect face and return bounding box"""
        if isinstance(image, np.ndarray):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        else:
            gray = np.array(image.convert('L'))
            
        faces = self.detector(gray, 1)
        if len(faces) == 0:
            return None
        return faces[0]
    
    def align_face(self, image, landmarks):
        """Align face based on eye positions"""
        if landmarks is None or len(landmarks) < 68:
            return image
            
        # Get left and right eye centers
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        
        # Calculate rotation angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate center and rotate
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), 
                                flags=cv2.INTER_CUBIC)
        return aligned
    
    def apply_clahe(self, image):
        """Apply CLAHE for contrast enhancement"""
        if not self.use_clahe:
            return image
            
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def preprocess(self, image_path, landmarks=None):
        """Full preprocessing pipeline"""
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE
        image = self.apply_clahe(image)
        
        # Align if landmarks provided
        if landmarks is not None:
            image = self.align_face(image, landmarks)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image

class AsymmetryAnalyzer:
    """NOVELTY: Advanced asymmetry analysis module"""
    
    def __init__(self):
        self.regions = {
            'eyebrows': (17, 27),
            'eyes': (36, 48),
            'nose': (27, 36),
            'mouth': (48, 68),
            'jaw': (0, 17)
        }
    
    def calculate_asymmetry_index(self, landmarks):
        """Calculate facial asymmetry index based on landmark distances"""
        if landmarks is None or len(landmarks) != 68:
            return 0.0
            
        # Mirror line is vertical through nose bridge (landmark 27, 28)
        mid_x = landmarks[27][0]
        
        asymmetry_scores = {}
        
        for region_name, (start, end) in self.regions.items():
            region_points = landmarks[start:end]
            left_dist = np.abs(region_points[:, 0] - mid_x)
            
            # Mirror points and calculate distance
            mirrored_x = 2 * mid_x - region_points[:, 0]
            right_points = np.column_stack([mirrored_x, region_points[:, 1]])
            
            # Calculate Euclidean distance between original and mirrored
            distances = np.linalg.norm(region_points - right_points, axis=1)
            asymmetry_scores[region_name] = np.mean(distances)
        
        # Weighted average (mouth and eyes are more important for palsy)
        weights = {'eyebrows': 0.15, 'eyes': 0.25, 'nose': 0.1, 
                  'mouth': 0.3, 'jaw': 0.2}
        
        total_asymmetry = sum(asymmetry_scores[r] * weights[r] 
                             for r in self.regions.keys())
        
        return total_asymmetry, asymmetry_scores

# Data augmentation with Albumentations
def get_train_augmentation(img_size=224):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, 
                          rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, 
                                  contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=img_size//20, 
                       max_width=img_size//20, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_augmentation(img_size=224):
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])