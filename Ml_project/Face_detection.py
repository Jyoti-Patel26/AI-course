"""
Enhanced Emotion Detection System using CNN
Features:
1. Real-time Webcam Emotion Detection
2. Comprehensive Model Evaluation & Visualization
3. Advanced Data Augmentation
"""

# ============================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 2. CONFIGURATION
# ============================================
class Config:
    # Paths
    TRAIN_PATH = "dataset/train"
    TEST_PATH = "dataset/test"
    MODEL_PATH = "emotion_model_best.h5"
    
    # Image parameters
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 15  # Increased for better training
    
    # Classes
    EMOTIONS = ['angry', 'happy', 'sad']  # Add more if available
    
    # Random seed for reproducibility
    SEED = 42
    
    # Real-time detection parameters
    WEBCAM_ID = 0  # 0 for built-in camera, 1 for external

np.random.seed(Config.SEED)
tf.random.set_seed(Config.SEED)

# ============================================
# 3. ENHANCED DATA LOADING WITH ADVANCED AUGMENTATION
# ============================================
def load_enhanced_data():
    """
    Load dataset with advanced augmentation techniques
    """
    # Enhanced training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,           # Random rotation
        width_shift_range=0.2,       # Horizontal shift
        height_shift_range=0.2,      # Vertical shift
        shear_range=0.2,             # Shear transformation
        zoom_range=0.2,              # Random zoom
        horizontal_flip=True,        # Horizontal flip
        fill_mode='nearest',         # Fill missing pixels
        brightness_range=[0.8, 1.2], # Brightness variation
        validation_split=0.2         # Split training/validation
    )
    
    # Testing data - only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data with augmentation
    train_data = train_datagen.flow_from_directory(
        Config.TRAIN_PATH,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=Config.SEED
    )
    
    # Load validation data
    val_data = train_datagen.flow_from_directory(
        Config.TRAIN_PATH,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=Config.SEED
    )
    
    # Load test data
    test_data = test_datagen.flow_from_directory(
        Config.TEST_PATH,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=Config.SEED
    )
    
    print("\n✅ Data Loading Summary:")
    print(f"   Training samples: {train_data.samples}")
    print(f"   Validation samples: {val_data.samples}")
    print(f"   Test samples: {test_data.samples}")
    print(f"   Classes: {train_data.class_indices}")
    
    return train_data, val_data, test_data

# ============================================
# 4. BUILD ENHANCED CNN MODEL
# ============================================
def build_enhanced_model(num_classes):
    """
    Build a CNN model with additional improvements:
    - Dropout layers to prevent overfitting
    - Batch Normalization for stable training
    - Increased depth for better feature extraction
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block (added for better feature extraction)
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

# ============================================
# 5. TRAIN MODEL WITH CALLBACKS
# ============================================
def train_model(model, train_data, val_data):
    """
    Train the model with advanced callbacks
    """
    # Compile model with Adam optimizer and learning rate scheduling
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            Config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train the model
    print("\n🚀 Starting Training...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Training Completed!")
    return history

# ============================================
# 6. COMPREHENSIVE MODEL EVALUATION
# ============================================
def evaluate_model(model, test_data, history):
    """
    Comprehensive model evaluation with metrics and visualizations
    """
    print("\n" + "="*50)
    print("📊 MODEL EVALUATION")
    print("="*50)
    
    # Test accuracy
    test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
    print(f"\n📈 Test Accuracy: {test_accuracy:.4f}")
    print(f"📉 Test Loss: {test_loss:.4f}")
    
    # Generate predictions
    test_data.reset()
    predictions = model.predict(test_data, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_data.classes
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(
        true_classes, 
        predicted_classes, 
        target_names=Config.EMOTIONS
    ))
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Training History Plot
    ax1 = axes[0, 0]
    ax1.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    ax1.set_title('Model Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss Plot
    ax2 = axes[0, 1]
    ax2.plot(history.history['loss'], label='Train Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    ax2.set_title('Model Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    ax3 = axes[0, 2]
    cm = confusion_matrix(true_classes, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=Config.EMOTIONS, 
                yticklabels=Config.EMOTIONS,
                ax=ax3)
    ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # 4. Bar Chart - Class-wise Accuracy
    ax4 = axes[1, 0]
    class_accuracy = []
    for i, emotion in enumerate(Config.EMOTIONS):
        idx = np.where(true_classes == i)[0]
        if len(idx) > 0:
            acc = np.mean(predicted_classes[idx] == i)
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0)
    
    bars = ax4.bar(Config.EMOTIONS, class_accuracy, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax4.set_title('Class-wise Accuracy', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy')
    ax4.set_ylim([0, 1])
    for bar, val in zip(bars, class_accuracy):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2%}', ha='center', fontweight='bold')
    
    # 5. Sample Predictions
    ax5 = axes[1, 1]
    # Show sample predictions
    sample_images = test_data[0][0][:6]
    sample_true = test_data[0][1][:6]
    sample_pred = model.predict(sample_images, verbose=0)
    sample_pred_classes = np.argmax(sample_pred, axis=1)
    
    for idx, (img, true, pred) in enumerate(zip(sample_images, sample_true, sample_pred_classes)):
        if idx < 6:
            ax = plt.subplot(2, 3, 4 + idx + 1) if idx < 3 else plt.subplot(2, 3, 7 + idx - 3) if idx < 6 else None
            if ax:
                ax.imshow(img)
                true_label = Config.EMOTIONS[np.argmax(true)]
                pred_label = Config.EMOTIONS[pred]
                color = 'green' if true_label == pred_label else 'red'
                ax.set_title(f'True: {true_label}\nPred: {pred_label}', 
                           fontsize=8, color=color)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_accuracy, test_loss

# ============================================
# 7. REAL-TIME WEBCAM EMOTION DETECTION
# ============================================
def real_time_detection(model):
    """
    Perform real-time emotion detection using webcam
    """
    print("\n" + "="*50)
    print("🎥 REAL-TIME EMOTION DETECTION")
    print("="*50)
    print("Instructions:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("  - Press 'h' to show/hide detection info")
    print("\nStarting webcam...")
    
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Open webcam
    cap = cv2.VideoCapture(Config.WEBCAM_ID)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    # Variables for performance
    fps = 0
    frame_count = 0
    start_time = datetime.now()
    show_info = True
    screenshot_count = 0
    
    # Create output directory for screenshots
    screenshot_dir = "screenshots"
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
    
    print("✅ Webcam started successfully!")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame")
            break
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            end_time = datetime.now()
            fps = 30 / (end_time - start_time).total_seconds()
            start_time = end_time
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess for model
            face_resized = cv2.resize(face_roi, Config.IMG_SIZE)
            face_normalized = face_resized / 255.0
            face_input = np.reshape(face_normalized, (1, 128, 128, 3))
            
            # Predict emotion
            prediction = model.predict(face_input, verbose=0)
            emotion_idx = np.argmax(prediction[0])
            emotion = Config.EMOTIONS[emotion_idx]
            confidence = prediction[0][emotion_idx] * 100
            
            # Define colors for different emotions
            emotion_colors = {
                'happy': (0, 255, 0),    # Green
                'sad': (255, 0, 0),      # Blue
                'angry': (0, 0, 255)     # Red
            }
            color = emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Display emotion and confidence
            if show_info:
                # Background for text
                cv2.rectangle(frame, (x, y-35), (x+w, y), color, -1)
                
                # Text
                text = f"{emotion.upper()} ({confidence:.1f}%)"
                cv2.putText(frame, text, (x+5, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display info panel
        if show_info:
            # Create info panel
            info_panel = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.rectangle(info_panel, (0, 0), (400, 100), (50, 50, 50), -1)
            
            # Add text
            cv2.putText(info_panel, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_panel, f"Faces Detected: {len(faces)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_panel, "Press 'h' to hide info, 's' to save", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Combine frames
            frame[0:100, 0:400] = info_panel
        
        # Display frame
        cv2.imshow('Real-Time Emotion Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n👋 Quitting webcam...")
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"{screenshot_dir}/emotion_detection_{timestamp}.png"
            cv2.imwrite(screenshot_path, frame)
            screenshot_count += 1
            print(f"📸 Screenshot saved: {screenshot_path}")
        elif key == ord('h'):
            show_info = not show_info
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Webcam closed. {screenshot_count} screenshots saved.")

# ============================================
# 8. SINGLE IMAGE PREDICTION
# ============================================
def predict_single_image(model, img_path):
    """
    Predict emotion from a single image file
    """
    print("\n" + "="*50)
    print("🖼️ SINGLE IMAGE PREDICTION")
    print("="*50)
    
    # Load and preprocess image
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Error: Could not load image from {img_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, Config.IMG_SIZE)
    img_normalized = img_resized / 255.0
    img_input = np.reshape(img_normalized, (1, 128, 128, 3))
    
    # Predict
    prediction = model.predict(img_input, verbose=0)
    emotion_idx = np.argmax(prediction[0])
    emotion = Config.EMOTIONS[emotion_idx]
    confidence = prediction[0][emotion_idx] * 100
    
    # Display results
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    
    # Add prediction text
    color = 'green' if confidence > 70 else 'orange'
    title = f"Prediction: {emotion.upper()}\nConfidence: {confidence:.2f}%"
    plt.title(title, fontsize=14, fontweight='bold', color=color)
    plt.axis('off')
    
    # Display confidence bars
    plt.figure(figsize=(8, 4))
    emotions = Config.EMOTIONS
    confidences = prediction[0] * 100
    
    bars = plt.bar(emotions, confidences, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.title('Emotion Probabilities', fontsize=14, fontweight='bold')
    plt.ylabel('Confidence (%)')
    plt.ylim([0, 100])
    
    for bar, conf in zip(bars, confidences):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{conf:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ Predicted Emotion: {emotion.upper()}")
    print(f"   Confidence: {confidence:.2f}%")
    
    return emotion, confidence

# ============================================
# 9. MAIN EXECUTION
# ============================================
def main():
    """
    Main function to run the emotion detection system
    """
    print("\n" + "="*50)
    print("😊 EMOTION DETECTION SYSTEM")
    print("="*50)
    print("Enhanced Features:")
    print("  1. Real-time Webcam Detection")
    print("  2. Comprehensive Model Evaluation")
    print("  3. Advanced Data Augmentation")
    print("="*50)
    
    # Check if model exists
    if os.path.exists(Config.MODEL_PATH):
        print(f"\n📦 Loading existing model from {Config.MODEL_PATH}")
        model = load_model(Config.MODEL_PATH)
        
        # Load data for evaluation
        _, _, test_data = load_enhanced_data()
        print("\n📊 Evaluating model performance...")
        test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
        print(f"   Test Accuracy: {test_accuracy:.4f}")
    else:
        print("\n🚀 Training new model...")
        # Load data
        train_data, val_data, test_data = load_enhanced_data()
        
        # Build model
        model = build_enhanced_model(len(Config.EMOTIONS))
        model.summary()
        
        # Train model
        history = train_model(model, train_data, val_data)
        
        # Evaluate model
        test_accuracy, test_loss = evaluate_model(model, test_data, history)
    
    # Interactive menu
    while True:
        print("\n" + "="*50)
        print("🎮 SELECT MODE")
        print("="*50)
        print("1. 🖼️  Single Image Prediction")
        print("2. 🎥  Real-time Webcam Detection")
        print("3. 📊  View Model Evaluation")
        print("4. ❌  Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            img_path = input("Enter image path: ").strip()
            if os.path.exists(img_path):
                predict_single_image(model, img_path)
            else:
                print(f"❌ Error: File not found - {img_path}")
        
        elif choice == '2':
            real_time_detection(model)
        
        elif choice == '3':
            # Reload test data for evaluation
            _, _, test_data = load_enhanced_data()
            test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
            print(f"\n📊 Model Performance:")
            print(f"   Test Accuracy: {test_accuracy:.4f}")
            print(f"   Test Loss: {test_loss:.4f}")
            
            # Show sample predictions
            print("\n🖼️ Sample Predictions:")
            test_data.reset()
            s
