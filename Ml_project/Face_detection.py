import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import time

# ===============================
# DATASET PATH
# ===============================

train_path = "dataset/train"
test_path = "dataset/test"

# ===============================
# DATA PREPROCESSING
# ===============================

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    train_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

# ===============================
# BUILD CNN MODEL
# ===============================

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(train_data.num_classes,activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Check if model already exists to avoid retraining
if not os.path.exists("emotion_model.h5"):
    # ===============================
    # TRAIN MODEL
    # ===============================
    
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=10,
        verbose=1
    )
    
    # ===============================
    # SAVE MODEL
    # ===============================
    
    model.save("emotion_model.h5")
    print("Model saved as emotion_model.h5")
    
    # Plot training history
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
else:
    # Load existing model
    model = tf.keras.models.load_model("emotion_model.h5")
    print("Model loaded from emotion_model.h5")

# ===============================
# NEW FEATURE: REAL-TIME WEBCAM EMOTION DETECTION
# ===============================

def real_time_detection():
    """Real-time emotion detection using webcam"""
    
    print("\n" + "="*50)
    print("REAL-TIME EMOTION DETECTION")
    print("="*50)
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    print("Press 'p' to pause/resume")
    print("="*50 + "\n")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get class labels
    labels = list(train_data.class_indices.keys())
    
    # Variables for FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    # Pause flag
    paused = False
    
    # Emotion colors for bounding box
    emotion_colors = {
        'happy': (0, 255, 0),      # Green
        'sad': (255, 0, 0),        # Blue
        'angry': (0, 0, 255),      # Red
        'surprise': (255, 255, 0), # Cyan
        'neutral': (255, 255, 255),# White
        'fear': (255, 0, 255),     # Magenta
        'disgust': (0, 255, 255)   # Yellow
    }
    
    # Face detector using OpenCV's Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Starting webcam...")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        if not paused:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                # Preprocess face for emotion detection
                face_resized = cv2.resize(face_roi, (128, 128))
                face_normalized = face_resized / 255.0
                face_input = np.reshape(face_normalized, (1, 128, 128, 3))
                
                # Predict emotion
                prediction = model.predict(face_input, verbose=0)
                predicted_class = labels[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                # Get color for bounding box
                color = emotion_colors.get(predicted_class, (255, 255, 255))
                
                # Draw bounding box around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Display emotion label with confidence
                label_text = f"{predicted_class}: {confidence:.1f}%"
                cv2.putText(frame, label_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add confidence bar
                bar_width = int(w * (confidence / 100))
                cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+15), color, -1)
                cv2.rectangle(frame, (x, y+h+5), (x+w, y+h+15), (128, 128, 128), 1)
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Display FPS and status
        status_text = "PAUSED" if paused else "LIVE"
        status_color = (0, 0, 255) if paused else (0, 255, 0)
        cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status_text}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Display instructions
        cv2.putText(frame, "q: quit | s: screenshot | p: pause", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show the frame
        cv2.imshow('Real-Time Emotion Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nExiting real-time detection...")
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"emotion_screenshot_{timestamp}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"Screenshot saved: {screenshot_path}")
        elif key == ord('p'):
            paused = not paused
            status = "paused" if paused else "resumed"
            print(f"Detection {status}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed")

# ===============================
# IMAGE UPLOAD AND PREDICTION
# ===============================

def image_upload_prediction():
    """Upload and predict emotion from a single image"""
    
    Tk().withdraw()
    
    img_path = askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    
    if not img_path:
        print("No image selected")
        return
    
    print("Selected image:", img_path)
    
    # ===============================
    # IMAGE PREPROCESSING
    # ===============================
    
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not load image")
        return
    
    img_resized = cv2.resize(img,(128,128))
    img_normalized = img_resized / 255.0
    img_input = np.reshape(img_normalized,(1,128,128,3))
    
    # ===============================
    # PREDICTION
    # ===============================
    
    prediction = model.predict(img_input, verbose=0)
    
    labels = list(train_data.class_indices.keys())
    
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    print(f"Predicted Emotion: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Display all class probabilities
    print("\nClass Probabilities:")
    for i, label in enumerate(labels):
        print(f"  {label}: {prediction[0][i]*100:.2f}%")
    
    # ===============================
    # DISPLAY RESULT
    # ===============================
    
    display_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12,5))
    
    # Show image with prediction
    plt.subplot(1,2,1)
    plt.imshow(display_img)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.1f}%", fontsize=14)
    plt.axis("off")
    
    # Show confidence bar chart
    plt.subplot(1,2,2)
    colors = ['green' if label == predicted_class else 'gray' for label in labels]
    bars = plt.bar(labels, prediction[0]*100, color=colors)
    plt.ylabel('Confidence (%)')
    plt.title('Emotion Prediction Probabilities')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for bar, val in zip(bars, prediction[0]*100):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# ===============================
# MAIN MENU
# ===============================

def main():
    """Main menu for selecting detection mode"""
    
    print("\n" + "="*50)
    print("EMOTION DETECTION SYSTEM")
    print("="*50)
    print("1. Upload Image for Emotion Detection")
    print("2. Real-Time Webcam Emotion Detection (NEW FEATURE!)")
    print("3. Exit")
    print("="*50)
    
    while True:
        try:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                image_upload_prediction()
                break
            elif choice == '2':
                real_time_detection()
                break
            elif choice == '3':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please select 1, 2, or 3")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
