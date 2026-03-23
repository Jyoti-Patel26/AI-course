import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# ===============================
# DATASET PATH
# ===============================

train_path = "dataset/train"
test_path = "dataset/test"

# ===============================
# DATA AUGMENTATION & PREPROCESSING
# ===============================

# Data augmentation for better generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale for test data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Load test data
test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Get class labels
food_labels = list(train_data.class_indices.keys())
print("\nFood Categories:", food_labels)
print(f"Number of classes: {train_data.num_classes}")

# ===============================
# BUILD CNN MODEL
# ===============================

model = Sequential()

# First Convolutional Block
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(2, 2))

# Second Convolutional Block
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Third Convolutional Block
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Fourth Convolutional Block
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Flatten and Dense Layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(train_data.num_classes, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n" + "="*50)
print("MODEL SUMMARY")
print("="*50)
print(model.summary())

# ===============================
# TRAIN MODEL WITH CALLBACKS
# ===============================

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Reduce learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=20,
    callbacks=[early_stop, reduce_lr]
)

# ===============================
# SAVE MODEL
# ===============================

model.save("food_model.h5")
print("\n✅ Model saved as food_model.h5")

# ===============================
# PLOT TRAINING HISTORY
# ===============================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Plot loss
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# ===============================
# TEST WITH NEW IMAGE
# ===============================

def predict_food(image_path):
    """
    Predict food type from an image
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))
    
    # Make prediction
    prediction = model.predict(img)
    predicted_class = food_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Display results
    display_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(display_img)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%", 
              fontsize=14, fontweight='bold')
    plt.axis("off")
    
    # Add probability bar chart
    plt.figure(figsize=(8, 4))
    probabilities = prediction[0] * 100
    bars = plt.bar(food_labels, probabilities, color='skyblue')
    plt.ylim(0, 100)
    plt.ylabel('Confidence (%)')
    plt.title('Prediction Probabilities')
    plt.xticks(rotation=45)
    
    # Color the highest bar
    bars[np.argmax(prediction)].set_color('green')
    
    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{prob:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return predicted_class, confidence

# ===============================
# EXAMPLE PREDICTION
# ===============================

# Replace with your image path
test_image_path = "test_food.jpg"  # Change this to your image path

if os.path.exists(test_image_path):
    print("\n" + "="*50)
    print("TESTING WITH SAMPLE IMAGE")
    print("="*50)
    result = predict_food(test_image_path)
    if result:
        predicted, confidence = result
        print(f"\n✅ Predicted Food: {predicted}")
        print(f"📊 Confidence: {confidence:.2f}%")
else:
    print(f"\n⚠️ Test image not found: {test_image_path}")
    print("To test with your own image, update the 'test_image_path' variable.")
