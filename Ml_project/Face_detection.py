#!/usr/bin/env python3
"""
Emotion Detection System with Real-Time Webcam Support
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Try importing required libraries with error handling
try:
    import cv2
    print("✓ OpenCV imported successfully")
except ImportError as e:
    print(f"✗ OpenCV import error: {e}")
    print("Installing OpenCV...")
    os.system("pip3 install opencv-python opencv-python-headless")
    import cv2

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    print("✓ TensorFlow imported successfully")
except ImportError as e:
    print(f"✗ TensorFlow import error: {e}")
    print("Installing TensorFlow...")
    os.system("pip3 install tensorflow")
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ===============================
# DATASET PATH
# ===============================

train_path = "dataset/train"
test_path = "dataset/test"

# Check if dataset exists
if not os.path.exists(train_path):
    print(f"\n⚠️  Warning: Dataset not found at {train_path}")
    print("Please create the following folder structure:")
    print("dataset/")
    print("  train/")
    print("    happy/")
    print("    sad/")
    print("    angry/")
    print("  test/")
    print("    happy/")
    print("    sad/")
    print("    angry/")
    print("\nOr download FER2013 dataset from Kaggle")
    
    # Create dummy data for testing
    create_dummy_data = input("\nCreate dummy data for testing? (y/n): ")
    if create_dummy_data.lower() == 'y':
        create_dummy_dataset()
    else:
        sys.exit(1)

# ===============================
# CREATE DUMMY DATASET FOR TESTING
# ===============================

def create_dummy_dataset():
    """Create dummy dataset for testing the code structure"""
    print("\nCreating dummy dataset for testing...")
    
    emotions = ['happy', 'sad', 'angry']
    
    for split in ['train', 'test']:
        for emotion in emotions:
            os.makedirs(f"dataset/{split}/{emotion}", exist_ok=True)
            
            # Create 5 dummy images for each category
            for i in range(5):
                # Create a random colored image
                dummy_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                
                # Add text to image based on emotion
                if emotion == 'happy':
                    dummy_img[60:68, 40:88] = [0, 255, 0]  # Green smile
                elif emotion == 'sad':
                    dummy_img[60:68, 40:88] = [255, 0, 0]  # Blue frown
                else:  # angry
                    dummy_img[60:68, 40:88] = [0, 0, 255]  # Red angry
                
                # Save image
                img_path = f"dataset/{split}/{emotion}/dummy_{i}.jpg"
                cv2.imwrite(img_path, dummy_img)
    
    print("✓ Dummy dataset created successfully!")

# ===============================
# DATA PREPROCESSING
# ===============================

print("\n" + "="*50)
print("Loading dataset...")
print("="*50)

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

print(f"\n✓ Found {train_data.num_classes} emotion classes:")
print(f"  {list(train_data.class_indices.keys())}")

# ===============================
# BUILD CNN MODEL
# ===============================

print("\n" + "="*50)
print("Building CNN Model...")
print("="*50)

model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(2, 2))

# Second Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Third Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Flatten Layer
model.add(Flatten())

# Dense Layers with Dropout
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(train_data.num_classes, activation='softmax'))

# Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ===============================
# TRAIN MODEL
# ===============================

if not os.path.exists("emotion_model.h5"):
    print("\n" + "="*50)
    print("Training Model...")
    print("="*50)
    
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=10,
        verbose=1
    )
    
    # Save model
    model.save("emotion_model.h5")
    print("\n✓ Model saved as emotion_model.h5")
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
else:
    print("\n✓ Loading existing model...")
    model = tf.keras.models.load_model("emotion_model.h5")
    print("✓ Model loaded successfully!")

# ===============================
# REAL-TIME WEBCAM DETECTION
# ===============================

def real_time_detection():
    """Real-time emotion detection using webcam"""
    
    print("\n" + "="*50)
    print("🎥 REAL-TIME EMOTION DETECTION")
    print("="*50)
    print("Controls:")
    print("  Press 'q' - Quit")
    print("  Press 's' - Save screenshot")
    print("  Press 'p' - Pause/Resume")
    print("="*50 + "\n")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        print("Try using image upload mode instead")
        return
    
    # Get class labels
    labels = list(train_data.class_indices.keys())
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Variables
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    paused = False
    
    print("✓ Webcam started! Show your face...")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        if not paused:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            for (x, y, w, h) in faces:
                # Extract face
                face_roi = frame[y:y+h, x:x+w]
                
                # Preprocess
                face_resized = cv2.resize(face_roi, (128, 128))
                face_normalized = face_resized / 255.0
                face_input = np.reshape(face_normalized, (1, 128, 128, 3))
                
                # Predict
                prediction = model.predict(face_input, verbose=0)
                predicted_class = labels[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                # Color coding
                if predicted_class == 'happy':
                    color = (0, 255, 0)
                elif predicted_class == 'sad':
                    color = (255, 0, 0)
                elif predicted_class == 'angry':
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 255)
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Display label
                label = f"{predicted_class}: {confidence:.1f}%"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Confidence bar
                bar_width = int(w * (confidence / 100))
                cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+15), color, -1)
                cv2.rectangle(frame, (x, y+h+5), (x+w, y+h+15), (128, 128, 128), 1)
        
        # FPS counter
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Display info
        status = "PAUSED" if paused else "LIVE"
        status_color = (0, 0, 255) if paused else (0, 255, 0)
        cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, "q:quit | s:screenshot | p:pause", 
                   (10, frame.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow('Emotion Detection System', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshot_{timestamp}.jpg", frame)
            print(f"📸 Screenshot saved!")
        elif key == ord('p'):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Webcam closed!")

# ===============================
# IMAGE UPLOAD PREDICTION
# ===============================

def image_upload_prediction():
    """Upload and predict emotion from image"""
    
    from tkinter import Tk, filedialog
    
    print("\n" + "="*50)
    print("📷 IMAGE UPLOAD MODE")
    print("="*50)
    
    # Create root window and hide it
    root = Tk()
    root.withdraw()
    
    # Open file dialog
    img_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    
    if not img_path:
        print("❌ No image selected")
        return
    
    print(f"✓ Selected: {os.path.basename(img_path)}")
    
    # Load and preprocess
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Could not load image")
        return
    
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    img_input = np.reshape(img_normalized, (1, 128, 128, 3))
    
    # Predict
    prediction = model.predict(img_input, verbose=0)
    labels = list(train_data.class_indices.keys())
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    print(f"\n🎯 Predicted Emotion: {predicted_class.upper()}")
    print(f"📊 Confidence: {confidence:.2f}%\n")
    
    print("📈 All Probabilities:")
    for i, label in enumerate(labels):
        bar = "█" * int(prediction[0][i] * 30)
        print(f"   {label:10s}: {prediction[0][i]*100:5.2f}% {bar}")
    
    # Display results
    display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(display_img)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.1f}%", fontsize=14)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    colors = ['green' if label == predicted_class else 'gray' for label in labels]
    bars = plt.bar(labels, prediction[0]*100, color=colors)
    plt.ylabel('Confidence (%)')
    plt.title('Emotion Probabilities')
    plt.ylim(0, 100)
    
    for bar, val in zip(bars, prediction[0]*100):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# ===============================
# MAIN MENU
# ===============================

def main():
    """Main menu"""
    
    print("\n" + "="*50)
    print("🎭 EMOTION DETECTION SYSTEM")
    print("="*50)
    print("1. 📷 Upload Image for Detection")
    print("2. 🎥 Real-Time Webcam Detection")
    print("3. ❌ Exit")
    print("="*50)
    
    while True:
        try:
            choice = input("\n👉 Select option (1-3): ").strip()
            
            if choice == '1':
                image_upload_prediction()
                break
            elif choice == '2':
                real_time_detection()
                break
            elif choice == '3':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice! Please select 1, 2, or 3")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

if __name__ == "__main__":
    main()
