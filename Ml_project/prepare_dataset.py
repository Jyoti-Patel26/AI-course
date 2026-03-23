import os
import shutil
import random
from sklearn.model_selection import train_test_split

# ===============================
# CONFIGURATION
# ===============================

# Path to your original dataset (all images organized by food type)
original_dataset_path = "original_food_dataset/"

# Paths for train/test split
train_path = "dataset/train/"
test_path = "dataset/test/"

# List of food categories
food_categories = ["pizza", "burger", "sushi", "pasta", "salad"]

# Split ratio
test_size = 0.2  # 20% for testing

# ===============================
# CREATE DIRECTORY STRUCTURE
# ===============================

# Create main directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Create category subdirectories
for category in food_categories:
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(test_path, category), exist_ok=True)

print("Directory structure created!")

# ===============================
# SPLIT DATASET
# ===============================

for category in food_categories:
    # Get all images in the original category folder
    category_path = os.path.join(original_dataset_path, category)
    
    if not os.path.exists(category_path):
        print(f"Warning: {category_path} not found. Skipping {category}")
        continue
    
    images = [f for f in os.listdir(category_path) 
              if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Split into train and test
    train_images, test_images = train_test_split(
        images, test_size=test_size, random_state=42
    )
    
    # Copy to train folder
    for img in train_images:
        src = os.path.join(category_path, img)
        dst = os.path.join(train_path, category, img)
        shutil.copy(src, dst)
    
    # Copy to test folder
    for img in test_images:
        src = os.path.join(category_path, img)
        dst = os.path.join(test_path, category, img)
        shutil.copy(src, dst)
    
    print(f"{category}: {len(train_images)} train, {len(test_images)} test")

print("\nDataset split completed!")
