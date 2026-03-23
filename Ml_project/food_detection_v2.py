import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Label, Button, Frame, messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# ===============================
# DATASET PATH
# ===============================

train_path = "dataset/train"
test_path = "dataset/test"

# ===============================
# CHECK IF MODEL EXISTS
# ===============================

model_path = "food_model.h5"

if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_data = train_datagen.flow_from_directory(
        train_path,
        target_size=(128,128),
        batch_size=32,
        class_mode='categorical'
    )
    food_labels = list(train_data.class_indices.keys())
    print("Model loaded successfully!")
else:
    print("Training new model...")
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
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_data = train_datagen.flow_from_directory(
        train_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )
    
    test_data = test_datagen.flow_from_directory(
        test_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )
    
    food_labels = list(train_data.class_indices.keys())
    
    # Build Model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(train_data.num_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nTraining model...")
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=20
    )
    
    model.save(model_path)
    print("Model saved!")

# ===============================
# PREDICTION FUNCTION
# ===============================

def predict_food(image_path):
    """Predict food type from an image"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))
    
    prediction = model.predict(img)
    predicted_class = food_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Get all probabilities
    probabilities = {food_labels[i]: prediction[0][i] * 100 
                    for i in range(len(food_labels))}
    
    return predicted_class, confidence, probabilities

# ===============================
# GUI APPLICATION
# ===============================

class FoodDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Food Detection System")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Title
        title = Label(root, text="🍕 Food Detection System 🍔", 
                     font=("Arial", 24, "bold"), 
                     bg='#f0f0f0', fg='#333')
        title.pack(pady=20)
        
        # Frame for image display
        self.image_frame = Frame(root, bg='white', relief='solid', bd=2)
        self.image_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        self.image_label = Label(self.image_frame, bg='white')
        self.image_label.pack(pady=10, padx=10, expand=True)
        
        # Upload Button
        self.upload_btn = Button(root, text="📁 Upload Food Image", 
                                 command=self.upload_image,
                                 font=("Arial", 14),
                                 bg='#4CAF50', fg='white',
                                 padx=20, pady=10,
                                 cursor='hand2')
        self.upload_btn.pack(pady=10)
        
        # Result Label
        self.result_label = Label(root, text="", 
                                  font=("Arial", 16, "bold"),
                                  bg='#f0f0f0', fg='#333')
        self.result_label.pack(pady=10)
        
        # Confidence Label
        self.confidence_label = Label(root, text="", 
                                      font=("Arial", 12),
                                      bg='#f0f0f0', fg='#666')
        self.confidence_label.pack()
        
    def upload_image(self):
        """Handle image upload and prediction"""
        file_path = filedialog.askopenfilename(
            title="Select a Food Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            # Display image
            img = Image.open(file_path)
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
            # Make prediction
            predicted, confidence, probabilities = predict_food(file_path)
            
            if predicted:
                # Update result
                self.result_label.configure(
                    text=f"🍽️ Predicted Food: {predicted.upper()} 🍽️",
                    fg='#4CAF50'
                )
                self.confidence_label.configure(
                    text=f"📊 Confidence: {confidence:.2f}%"
                )
                
                # Show probability chart
                self.show_probability_chart(probabilities, predicted)
            else:
                messagebox.showerror("Error", "Could not process image!")
    
    def show_probability_chart(self, probabilities, predicted_class):
        """Display probability bar chart"""
        plt.figure(figsize=(8, 5))
        categories = list(probabilities.keys())
        values = list(probabilities.values())
        
        bars = plt.bar(categories, values, color='skyblue')
        bars[categories.index(predicted_class)].set_color('green')
        
        plt.ylim(0, 100)
        plt.ylabel('Confidence (%)', fontsize=12)
        plt.title('Prediction Probabilities', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# ===============================
# RUN APPLICATION
# ===============================

if __name__ == "__main__":
    root = Tk()
    app = FoodDetectionApp(root)
    root.mainloop()
