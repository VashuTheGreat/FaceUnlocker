from keras.models import load_model
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import pickle
import numpy as np

import os
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cropper = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)

class Model:
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def get_features(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cropper.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            raise ValueError("No face detected in the image.")

        x, y, w, h = faces[0]
        face = rgb_img[y:y+h, x:x+w]

        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face, axis=0)        
        face = face.astype('float32')              
        face = preprocess_input(face)

        features = self.model.predict(face)
        return features.flatten().reshape(-1, 1)

def cosine_similarity_score(features1, features2):
    return cosine_similarity(features1, features2)[0][0]

model = Model('model.h5')

while True:
    ret, frame = cap.read()

    if ret:
        cv2.imwrite("frame.jpg", frame)
        try:
            features = model.get_features("frame.jpg")
        except ValueError as e:
            print("No face detected in the frame,Retrying...")
            continue    
        if not os.path.exists("my_feature.pkl"):
            try:
                my_feature = model.get_features("my_img.jpg")
            except ValueError as e:
                print("No face detected in the reference image, please provide a valid image.")
                break    
            with open("my_feature.pkl", "wb") as f:
                pickle.dump(my_feature, f)
        with open('my_feature.pkl', 'rb') as f:
            my_feature = pickle.load(f)      
        score = cosine_similarity_score(my_feature, features)
        if score > 0.5:
            print("✅ Face matched unlocked")
            break
        else:
            print("❌ Face not matched, try again")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cropper.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(face):
            x, y, w, h = face[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        else:
            # print("No face detected")  
            pass
        cv2.imshow("Frame", frame)      
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
