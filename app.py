import cv2
import numpy as np
from retinaface import RetinaFace
from facenet_pytorch import InceptionResnetV1
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle

# Initialize the FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load known embeddings from a file or create a new dictionary
try:
    with open("embedding_database.pkl", "rb") as f:
        embedding_database = pickle.load(f)
except FileNotFoundError:
    embedding_database = {}

# Function to recognize the face
def recognize_face(new_embedding, embedding_database, threshold=0.6):
    min_distance = float("inf")
    identity = "Unknown"
    
    for name, embedding in embedding_database.items():
        # Calculate Euclidean distance
        distance = np.linalg.norm(new_embedding - embedding)
        
        if distance < min_distance:
            min_distance = distance
            identity = name
    
    if min_distance < threshold:
        return identity
    else:
        return "Unknown"

# Main function for face detection and recognition
def detect_and_recognize_face(image_path=None):
    if image_path:
        # Load the image from the given path
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image at {image_path}")
            return
    else:
        # Use the webcam to capture an image
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to access the camera")
            return
        ret, img = cap.read()
        cap.release()
        if not ret:
            print("Error: Unable to capture image from camera")
            return

    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(img)

    if isinstance(faces, dict):
        for key in faces:
            face = faces[key]
            facial_area = face["facial_area"]
            landmarks = face["landmarks"]

            # Crop the face using the detected bounding box
            x1, y1, x2, y2 = facial_area
            face_crop = img[y1:y2, x1:x2]

            # Resize the cropped face to a standard size (e.g., 160x160 pixels)
            face_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_AREA)

            # Convert to PIL Image for compatibility with FaceNet
            face_img = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))

            # Preprocess face image for FaceNet
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            face_tensor = preprocess(face_img).unsqueeze(0).to(device)

            # Get the embedding for the face
            with torch.no_grad():
                embedding = model(face_tensor).cpu().numpy()

            # Recognize the face if not enrolling
            name = recognize_face(embedding, embedding_database)

            if name == "Unknown":
                print("Face not recognized.")
                response = input("Do you want to add this face to the database? (yes/no): ").strip().lower()
                if response == 'yes':
                    enroll_name = input("Enter the name of the person: ").strip()
                    embedding_database[enroll_name] = embedding
                    with open("embedding_database.pkl", "wb") as f:
                        pickle.dump(embedding_database, f)
                    print(f"Face enrolled as '{enroll_name}'")
                else:
                    print("Face not added to the database.")
            else:
                print(f"Recognized as {name}")

            # Display the name on the original image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Optionally, draw landmarks
            for landmark_key in landmarks:
                landmark_point = landmarks[landmark_key]
                cv2.circle(img, tuple(map(int, landmark_point)), 2, (0, 0, 255), -1)

    # Display the output with detected faces
    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = None  # Set to "path/to/your/image.jpg" if you want to use an image file
    detect_and_recognize_face(image_path)
