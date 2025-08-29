import os
import time
import json
import pickle
import shutil
import random
import string
from collections import Counter
from typing import Dict, Any, Optional
from datetime import datetime
import cv2
import torch
import numpy as np
from scipy.spatial.distance import cosine
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from kafka.errors import KafkaError

# Local imports
from src.kafka.producer import response_writer, HrmsEmployeeRegisterResponse
from src.kafka.consumer import is_recieving_frame
from src.face_registration import face_detect_with_kafka
from src.get_embedding import get_embedding


# Configuration
class Config:
    FACE_EMBEDDING_PATH = r"src\embedding_dir\face_embeddings.pkl"
    
    # Directory paths
    SAVE_DIR = "Data Uploaded"
    VALIDATION_FACE_DIR = 'face_validation'
    
    # Thresholds
    EMBEDDING_THRESHOLD = 0.09  # Cosine similarity threshold for face matching
    IMAGES_PER_ANGLE = 5       # Number of images to capture per angle
    MAX_REGISTRATION_IMAGES = 120  # Maximum images to capture during registration
    VALIDATION_IMAGES = 9       # Number of validation images to capture
    
    # Kafka
    KAFKA_TOPIC = "hrms.employee.register.response"
    
    @staticmethod
    def setup_directories():
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
        os.makedirs(Config.VALIDATION_FACE_DIR, exist_ok=True)

# Initialize configuration
Config.setup_directories()

class FaceRegistrationSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.facenet_model = self._load_facenet_model()

        
        # State tracking
        self.registration_state: Dict[str, Dict[str, Any]] = {}
        self.last_time = time.time()
        
  

    def _load_facenet_model(self):
        """Load the FaceNet model for embeddings"""
        model = InceptionResnetV1(
            pretrained='vggface2',
            classify=False
        ).eval().to(self.device)
        return model




    def _initialize_user_state(self, user_key: str):
        """Initialize tracking state for a new user"""
        if user_key not in self.registration_state:
            self.registration_state[user_key] = {
                "angle_counts": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0,"6":0,"7":0,"8":0},
                "success_sent": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0,"6":0,"7":0,"8":0},
                "total_images": 0,
                "is_duplicate": False,
                "validation_done": False
            }

    def _save_validation_images(self, name: str, emp_code: str, frame: np.ndarray) -> bool:
        """Save validation images for face matching"""
        user_key = f"{name}_{emp_code}"
        validation_dir = os.path.join(Config.VALIDATION_FACE_DIR, user_key)
        os.makedirs(validation_dir, exist_ok=True)
        
        # Capture validation images
        for i in range(Config.VALIDATION_IMAGES):
            face_img = face_detect_with_kafka(
                person_name=name,
                person_id=emp_code,
                faces_dir=validation_dir,
                img_count_thresh=i+1,
                frame=frame
            )
            
            if face_img is None:
                continue
                
            secret_code = "".join(random.choices(string.ascii_letters + string.digits, k=4))
            img_path = os.path.join(validation_dir, f"{user_key}_{secret_code}.png")
            cv2.imwrite(img_path, face_img)
            print(f"Saved validation image: {img_path}")
            
        return True

    def _check_existing_user(self, name: str, emp_code: str) -> tuple:
        """Check if user already exists in the system"""
        user_key = f"{name}_{emp_code}"
        validation_dir = os.path.join(Config.VALIDATION_FACE_DIR, user_key)
        
        # Get embeddings for new user
        new_user_embs = get_embedding(
            dataset_dir=Config.VALIDATION_FACE_DIR,
            device=self.device,
            validation_embedding=True
        )
        
        # Clean up validation directory
        try:
            if os.path.exists(validation_dir):
                shutil.rmtree(validation_dir)
        except Exception as e:
            print(f"Warning: Could not delete validation directory: {str(e)}")

        # Load registered embeddings
        try:
            with open(Config.FACE_EMBEDDING_PATH, 'rb') as f:
                registered_user_embs = pickle.load(f)
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return "unidentified", None

        # Compare embeddings
        best_match, best_score = self._compare_embeddings(new_user_embs, registered_user_embs)
        
        is_registered = best_score < Config.EMBEDDING_THRESHOLD
        return best_match if is_registered else "unidentified", is_registered

    def _compare_embeddings(self, new_embs: dict, registered_embs: dict) -> tuple:
        """Compare face embeddings to find best match"""
        best_score = float("inf")
        best_match = "unidentified"
        results = []
        
        for new_user, embeddings in new_embs.items():
            for new_embedding in embeddings:
                for reg_user, reg_embeddings in registered_embs.items():
                    for reg_embedding in reg_embeddings:
                        distance = cosine(new_embedding.flatten(), reg_embedding.flatten())
                        if distance < best_score:
                            best_score = distance
                            best_match = reg_user
                
                results.append({
                    "Emp": best_match if best_score < Config.EMBEDDING_THRESHOLD else "unidentified",
                    "Dist": best_score
                })
        
        if not results:
            return "unidentified", float("inf")
            
        # Get most common match
        emp_counter = Counter(entry['Emp'] for entry in results)
        if not emp_counter:
            return "unidentified", float("inf")
            
        final_match = emp_counter.most_common(1)[0][0]
        return final_match, best_score

    def _process_new_user(self, name: str, emp_code: str, frame: np.ndarray,image_angle:str):
        """Process registration for a new user"""
        user_key = f"{name}_{emp_code}"
        user_state = self.registration_state[user_key]
        
        # Detect face
        face_img = face_detect_with_kafka(
            person_name=name,
            person_id=emp_code,
            faces_dir=Config.VALIDATION_FACE_DIR,
            img_count_thresh=user_state["total_images"] + 1,
            frame=frame
        )
        
        if face_img is None:
            return
            
        # print(image_angle)
        # Check if we need more images of this angle
        if user_state["angle_counts"][image_angle] < Config.IMAGES_PER_ANGLE:
            self._save_registration_image(name, emp_code, face_img, image_angle)
            user_state["angle_counts"][image_angle] += 1
            user_state["total_images"] += 1
            
            print(f"Progress for {user_key}: {user_state['angle_counts']}")
            
            # Send success message if angle quota met
            if user_state["angle_counts"][image_angle] == Config.IMAGES_PER_ANGLE:
                self._send_angle_success_message(name, emp_code, image_angle)
                
        # Check if registration is complete
        if all(count >= Config.IMAGES_PER_ANGLE 
               for count in user_state["angle_counts"].values()):
            print(f"Registration complete for {user_key}. Generating embeddings...")
            _ = get_embedding(
                dataset_dir=Config.SAVE_DIR,
                device=self.device,
                validation_embedding=False
            )
            user_state["is_duplicate"] = True  # Mark as registered

    def _save_registration_image(self, name: str, emp_code: str, 
                               face_img: np.ndarray, angle_label: str):
        """Save registration image to disk"""
        user_dir = os.path.join(Config.SAVE_DIR, f"{name}_{emp_code}")
        os.makedirs(user_dir, exist_ok=True)
        
        secret_code = "".join(random.choices(string.ascii_letters + string.digits, k=4))
        img_path = os.path.join(user_dir, 
                               f"{name}_{emp_code}_{secret_code}_{angle_label}.png")
        cv2.imwrite(img_path, face_img)

        self._send_angle_success_message(name,emp_code,angle_label, everyimginfo=f"{angle_label}.{self.registration_state[f'{name}_{emp_code}']['angle_counts'][angle_label]}")
        print("---->   ",f"{angle_label}.{self.registration_state[f'{name}_{emp_code}']['angle_counts'][angle_label]}")  
        print(f"Saved registration image: {img_path}")

    def _send_angle_success_message(self, name: str, emp_code: str, angle_label: str,everyimginfo=None):
        """Send Kafka message when an angle is successfully captured"""
        response = HrmsEmployeeRegisterResponse(
            userName=name,
            empCode=emp_code,
            statusCode="Created",
            message=f"Image Details - {f'Done - {angle_label}' if everyimginfo is None else everyimginfo}",
            faceAngle=f"{angle_label}.{self.registration_state[f'{name}_{emp_code}']['angle_counts'][angle_label]}",
            timeStamp=datetime.now().isoformat()
        )
        
        try:
            response_writer(message=json.dumps(response.__dict__).encode('ascii'))
        except KafkaError as e:
            print(f"Failed to send Kafka message: {str(e)}")

    def _send_duplicate_user_message(self, name: str, emp_code: str, matched_user: str):
        """Send Kafka message for duplicate user"""
        response = HrmsEmployeeRegisterResponse(
            userName=name,
            empCode=emp_code,
            statusCode="duplicated user",
            message=f"User already registered as: {matched_user}",
            faceAngle="not fetched images",
            timeStamp=datetime.now().isoformat()
        )
        
        try:
            response_writer(message=json.dumps(response.__dict__).encode('ascii'))
        except KafkaError as e:
            print(f"Failed to send Kafka message: {str(e)}")

    def run(self):
        """Main processing loop"""
        print("Face registration system started. Waiting for Kafka messages...")
        
        while True:
            # try:
            # Get next frame from Kafka
            name, emp_code, frame, image_angle = is_recieving_frame()
            print(name,emp_code,image_angle)
            if frame is None:
                continue
                
            user_key = f"{name}_{emp_code}"
            self._initialize_user_state(user_key)
            user_state = self.registration_state[user_key]
            
            # Validation phase
            if not user_state["validation_done"]:
                self._save_validation_images(name, emp_code, frame)
                user_state["validation_done"] = True
                
                # Check if user exists
                match_result, is_registered = self._check_existing_user(name, emp_code)
                print(f"Match result: {match_result}, Registered: {is_registered}")
                
                if is_registered:
                    user_state["is_duplicate"] = True
                    self._send_duplicate_user_message(name, emp_code, match_result)
                    continue
            
            # Registration phase
            if not user_state["is_duplicate"]:
                if user_state["total_images"] < Config.MAX_REGISTRATION_IMAGES:
                    self._process_new_user(name, emp_code, frame,image_angle)
                else:
                    print(f"Maximum registration images reached for {user_key}")
                    user_state["is_duplicate"] = True
                        


if __name__ == "__main__":
    registration_system = FaceRegistrationSystem()
    registration_system.run()