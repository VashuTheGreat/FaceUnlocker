import cv2
import os
import pickle
import numpy as np
from keras.models import load_model
from keras_vggface.utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3



def speak(text):
    engine = pyttsx3.init()  # new instance every time
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()


# Haar Cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cropper = cv2.CascadeClassifier(cascade_path)


# ✅ Model class optimized to accept frame directly
class Model:
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def get_features_from_frame(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cropper.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            raise ValueError("No face detected in the frame.")

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

# ✅ Main function optimized
def call():
    speak("This machine is locked")

    cap = cv2.VideoCapture(0)

    # Preload reference features if exist
    my_feature = None
    if os.path.exists("my_feature.pkl"):
        with open('my_feature.pkl', 'rb') as f:
            my_feature = pickle.load(f)
    else:
        if not os.path.exists("my_img.jpg"):
            print("Reference image my_img.jpg not found!")
            speak("Reference image not found")
            cap.release()
            return False
        ref_img = cv2.imread("my_img.jpg")
        my_feature = model.get_features_from_frame(ref_img)
        with open("my_feature.pkl", "wb") as f:
            pickle.dump(my_feature, f)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            features = model.get_features_from_frame(frame)
        except ValueError:
            print("No face detected in the frame, retrying...")
            continue    

        score = cosine_similarity_score(my_feature, features)
        if score > 0.5:
            print("✅ Face matched unlocked")
            speak('Face Matched .... Unlocking ...')
            cap.release()
            cv2.destroyAllWindows()
            return True
        else:
            print("❌ Face not matched, try again")
            speak('Face not matched, try again')

        # Draw face box if detected
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cropper.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(face):
            x, y, w, h = face[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
        cv2.imshow("Frame", frame)      
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False  

if __name__ == "__main__":
    val = call()
