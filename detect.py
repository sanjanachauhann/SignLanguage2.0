# from tensorflow.python.keras.models import load_model
import cv2
import numpy as np
import os
import mediapipe as mp
from keras.models import load_model
from matplotlib import pyplot as plt

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

actions = np.array([
     'B', 'can' , 
     'again' , 'careful' , 'do' , 'hello' , 'help' , 'listen' , 'more' ])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_face_box(image, results):
    face_landmarks = results.face_landmarks.landmark
    left_top_x = min([landmark.x for landmark in face_landmarks])
    left_top_y = min([landmark.y for landmark in face_landmarks])
    right_bottom_x = max([landmark.x for landmark in face_landmarks])
    right_bottom_y = max([landmark.y for landmark in face_landmarks])

    # Extend the box a little bit
    left_top_x = max(0, int(left_top_x * image.shape[1]) - 10)
    left_top_y = max(0, int(left_top_y * image.shape[0]) - 10)
    right_bottom_x = min(image.shape[1], int(right_bottom_x * image.shape[1]) + 10)
    right_bottom_y = min(image.shape[0], int(right_bottom_y * image.shape[0]) + 10)

    # Draw pink rectangle
    cv2.rectangle(image, (left_top_x, left_top_y), (right_bottom_x, right_bottom_y), (180, 105, 255), 2)

    return left_top_x, left_top_y, right_bottom_x, right_bottom_y

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # Calculate text width
        text_width, _ = cv2.getTextSize(actions[num], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int((prob * 100) *text_width[0] + 10 ), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame

def prob_viz_modified(res, actions):
    action_index = np.argmax(res)
    print("Action Detected:", actions[action_index])
    return actions[action_index]

colors = [(245, 117, 16),   # Orange
          (0, 255, 0),       # Green
          (255, 0, 0),       # Red
          (0, 0, 255),       # Blue
          (255, 255, 0),     # Yellow
          (128, 0, 128),     # Purple
          (255, 165, 0),     # Orange
          (0, 128, 128),     # Teal
          (255, 20, 147),    # Deep Pink
          (0, 255, 255),     # Cyan
          (0, 0, 0),         # Black
          (255, 255, 255),   # White
          (128, 128, 128),   # Gray
          (255, 192, 203),   # Pink
          (0, 0, 128),       # Navy
          (128, 0, 0),       # Maroon
          (0, 128, 0),       # Olive
          (128, 128, 0),     # Olive Green
          (128, 0, 128),     # Purple
          (0, 128, 128),     # Teal
          (255, 69, 0)       # Red-Orange
          ]

# Load the model
path = 'model/action.keras'
try:
    model = load_model(path)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading the model:", e)

# actions = np.array(['hello'])

# New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        #Draw box 
        left_top_x , right_bottom_y , right_bottom_x, right_bottom_y = draw_face_box(image , results)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-20:]

        if len(sequence) == 20:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            # Update sentence based on recognition result
            if np.unique(predictions[-10:])[0] == np.argmax(res) and res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        # sentence.append(actions[np.argmax(res)])
                        sentence.clear()
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Visualize probabilities
            image = prob_viz(res, actions, image, colors) 
            
            # Get the detected action using prob_viz_modified function
            detected_action = prob_viz_modified(res, actions)
            box_image = frame.copy()
            
    # Display the detected word at the bottom of the face box using the detected_action
            cv2.putText(box_image, detected_action, (left_top_x + 10, right_bottom_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Sign language detected' , box_image)        

            # Show the frame
            cv2.imshow(' Detection Analysis', image)
       
       
        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
