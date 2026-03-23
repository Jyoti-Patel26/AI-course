import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

# ===============================
# DATA SPLIT EXAMPLE
# ===============================

print("="*50)
print("TRAIN-TEST SPLIT DEMONSTRATION")
print("="*50)

# Example with sample data
print("\n1. Basic Example with Numbers:")
nums = [1, 2, 3, 44, 5, 6, 7, 8]
train_nums, test_nums = train_test_split(nums, test_size=0.2, random_state=42)
print(f"Original: {nums}")
print(f"Training: {train_nums}")
print(f"Testing: {test_nums}")

# Example with Iris dataset
print("\n2. Example with Iris Dataset:")
from sklearn.datasets import load_iris

data = load_iris()
x = data.data  # features
y = data.target  # labels

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    train_size=0.8,
    random_state=40
)

print(f"Training features shape: {x_train.shape}")
print(f"Testing features shape: {x_test.shape}")

# ===============================
# FOOD DATASET SPLIT FUNCTION
# ===============================

def split_food_dataset(original_path, train_path, test_path, test_size=0.2):
    """
    Split food dataset into train and test directories
    
    Args:
        original_path: Path to original dataset (with category folders)
        train_path: Path to save training images
        test_path: Path to save testing images
        test_size: Proportion of data for testing (default: 0.2)
    """
    
    # Get all categories
    categories = [f for f in os.listdir(original_path) 
                 if os.path.isdir(os.path.join(original_path, f))]
    
    print(f"\nFound categories: {categories}")
    
    for category in categories:
        category_path = os.path.join(original_path, category)
        images = [f for f in os.listdir(category_path) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Split images
        train_imgs, test_imgs = train_test_split(
            images, test_size=test_size, random_state=42
        )
        
        # Create category directories if they don't exist
        os.makedirs(os.path.join(train_path, category), exist_ok=True)
        os.makedirs(os.path.join(test_path, category), exist_ok=True)
        
        # Copy images
        for img in train_imgs:
            src = os.path.join(category_path, img)
            dst = os.path.join(train_path, category, img)
            shutil.copy(src, dst)
        
        for img in test_imgs:
            src = os.path.join(category_path, img)
            dst = os.path.join(test_path, category, img)
            shutil.copy(src, dst)
        
        print(f"{category}: {len(train_imgs)} train, {len(test_imgs)} test")
    
    print("\n✅ Dataset split completed!")
    return categories

# ===============================
# UNCOMMENT TO USE:
# ===============================

# original_dataset = "original_food_dataset/"
# train_dir = "dataset/train/"
# test_dir = "dataset/test/"
# 
# categories = split_food_dataset(original_dataset, train_dir, test_dir, test_size=0.2)

print("\n" + "="*50)
print("To split your food dataset, uncomment the code above")
print("and set the correct paths.")
print("="*50)
