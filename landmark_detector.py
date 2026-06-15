# """
# Facial landmark detection using dlib
# """
# import dlib
# import numpy as np
# import cv2
# from pathlib import Path
# import os

# class LandmarkDetector:
#     def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
#         self.detector = dlib.get_frontal_face_detector()
        
#         # Try multiple possible locations
#         possible_paths = [
#             predictor_path,  # Current directory
#             Path(__file__).parent.parent / predictor_path,  # Project root
#             Path.cwd() / predictor_path,  # Working directory
#             Path("C:/Users/Astha Paika/Desktop/facial_paralysis_detection") / predictor_path,  # Your specific path
#         ]
        
#         found_path = None
#         for path in possible_paths:
#             path = Path(path)
#             print(f"Checking: {path.absolute()} - Exists: {path.exists()}")
#             if path.exists():
#                 # Check file size (should be ~97MB)
#                 size_mb = path.stat().st_size / (1024 * 1024)
#                 print(f"  File size: {size_mb:.1f} MB")
#                 if size_mb > 90:  # Valid file should be >90MB
#                     found_path = str(path.absolute())
#                     break
        
#         if found_path is None:
#             raise FileNotFoundError(
#                 f"\n{'='*60}\n"
#                 f"ERROR: dlib model file not found or invalid!\n"
#                 f"{'='*60}\n"
#                 f"Searched in:\n" + 
#                 "\n".join([f"  - {p.absolute()}" for p in possible_paths]) +
#                 f"\n\nPlease ensure 'shape_predictor_68_face_landmarks.dat' exists "
#                 f"and is ~97 MB (not the .bz2 file).\n"
#                 f"{'='*60}"
#             )
        
#         print(f"\nLoading dlib model from: {found_path}")
#         try:
#             self.predictor = dlib.shape_predictor(found_path)
#             print("✓ Model loaded successfully!")
#         except RuntimeError as e:
#             print(f"\n{'='*60}")
#             print(f"ERROR loading model: {e}")
#             print(f"The file might be corrupted. Please re-download.")
#             print(f"{'='*60}")
#             raise
        
#     def detect(self, image):
#         """Detect 68 facial landmarks"""
#         if isinstance(image, str) or isinstance(image, Path):
#             image = cv2.imread(str(image))
#             if image is None:
#                 return None
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         elif hasattr(image, 'convert'):  # PIL Image
#             image = np.array(image)
            
#         gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#         faces = self.detector(gray, 1)
        
#         if len(faces) == 0:
#             return None
            
#         face = faces[0]
#         landmarks = self.predictor(gray, face)
        
#         # Convert to numpy array
#         coords = np.zeros((68, 2), dtype=np.float32)
#         for i in range(68):
#             coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
            
#         return coords
    
#     def detect_batch(self, image_paths, save_dir):
#         """Process batch and save landmarks"""
#         save_dir = Path(save_dir)
#         save_dir.mkdir(parents=True, exist_ok=True)
        
#         results = {}
#         for img_path in image_paths:
#             landmarks = self.detect(img_path)
#             save_path = save_dir / f"{Path(img_path).stem}.npy"
            
#             if landmarks is not None:
#                 np.save(save_path, landmarks)
#                 results[img_path] = landmarks
#             else:
#                 np.save(save_path, np.zeros((68, 2)))
#                 results[img_path] = None
                
#         return results

"""
Facial landmark detection using dlib
"""
import dlib
import numpy as np
import cv2
from pathlib import Path
import warnings
import os

class LandmarkDetector:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        
        # Check if model file exists
        predictor_file = Path(predictor_path)
        
        # Try multiple possible locations
        possible_paths = [
            predictor_file,
            Path(__file__).parent.parent / predictor_path,  # Project root
            Path(__file__).parent.parent.parent / predictor_path,  # One level up
            Path(os.getcwd()) / predictor_path,  # Current working directory
        ]
        
        found_path = None
        for p in possible_paths:
            p = Path(p)  # Ensure it's a Path object
            print(f"Checking: {p.absolute()} - Exists: {p.exists()}")
            if p.exists():
                # Check file size (should be ~95 MB, not 0.3 MB)
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"  File size: {size_mb:.1f} MB")
                if size_mb > 10:  # Valid model should be >10 MB
                    found_path = str(p)
                    break
        
        if found_path is None:
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"ERROR: dlib model file not found or invalid!\n"
                f"{'='*60}\n"
                f"Searched in:\n" +
                "\n".join([f"  - {Path(p).absolute()}" for p in possible_paths]) +
                f"\n\nPlease download the model from:\n"
                f"http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n\n"
                f"Extract it and place the .dat file (should be ~95 MB) in:\n"
                f"  {Path(__file__).parent.parent.absolute()}\n"
                f"{'='*60}"
            )
        
        print(f"Loading dlib model from: {found_path}")
        self.predictor = dlib.shape_predictor(found_path)
        print("✓ Model loaded successfully")
        
    def detect(self, image):
        """Detect 68 facial landmarks"""
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif hasattr(image, 'convert'):  # PIL Image
            image = np.array(image)
            
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray, 1)
        
        if len(faces) == 0:
            return None
            
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Convert to numpy array
        coords = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
            
        return coords
    
    def detect_batch(self, image_paths, save_dir):
        """Process batch and save landmarks"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        for img_path in image_paths:
            save_path = save_dir / f"{Path(img_path).stem}.npy"
            results[img_path] = save_path
            
            if not save_path.exists():
                landmarks = self.detect(img_path)
                if landmarks is not None:
                    np.save(save_path, landmarks)
                else:
                    np.save(save_path, np.zeros((68, 2), dtype=np.float32))
                
        return results