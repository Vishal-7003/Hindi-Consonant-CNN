# Handwriting-Recognition-and-Classification-of-Hindi-Consonants

<h2 align="center">CNN-based Multiclass Classification (36 Classes)</h2>
<br>
<br>
<img width="1536" height="1024" alt="cnn1" src="https://github.com/user-attachments/assets/ff3d81d9-97ab-403a-8b22-b21e4d2eff9d" />
<br>
<br>
<br>
<br>

# 2000 images for each class (36 classes in total)
<img width="1024" height="1024" alt="pic3" src="https://github.com/user-attachments/assets/ef320ba1-d386-4cbe-aa90-04903a9fa602" />

# **Tech Stack Used** :
<br>

1. Python
2. TensorFlow / Keras
3. NumPy
4. Matplotlib
5. Scikit-learn
   
<br>

# **WorkFlow**

## Loading and Unzipping dataset

<img width="860" height="645" alt="image" src="https://github.com/user-attachments/assets/b075ee6d-62c3-40b0-987d-c6f4c4a3a4dd" />

<br>

## Train,Validation and Test Data Split
```python
import tensorflow as tf

BATCH_SIZE = 64
IMG_SIZE = (32, 32)

train_ds = tf.keras.utils.image_dataset_from_directory(
    "/content/drive/MyDrive/dataset/train",
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "/content/drive/MyDrive/dataset/train",
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "/content/drive/MyDrive/dataset/test",
    image_size=(32, 32),
    color_mode="grayscale",
    batch_size=64,
    shuffle=False
)
```
## Model Architecture :
```python
model = tf.keras.Sequential([

    #  Normalize pixels inside the model
    tf.keras.layers.Rescaling(1./255, input_shape=(32, 32, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    # Dropout to prevent overfitting
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(36, activation='softmax')
])
```

## Model Training:
<img width="1217" height="532" alt="image" src="https://github.com/user-attachments/assets/601620d1-9699-4fd9-920d-4b44fce3cf5a" />

## Prediction :
```python
import matplotlib.pyplot as plt
import numpy as np

# Take one batch from test data
for images, labels in test_ds.take(1):
    image = images[0]          # first image
    true_label = labels[0]     # true label

    # Add batch dimension and predict
    prediction = model.predict(tf.expand_dims(image, axis=0))
    predicted_label = np.argmax(prediction)

    break

plt.figure(figsize=(3, 3))
plt.imshow(image.numpy().squeeze(), cmap="gray")
plt.axis("off")

plt.title(
    f"Actual: {class_names[true_label]}\nPredicted: {class_names[predicted_label]}"
)
plt.show()

```
<img width="318" height="363" alt="image" src="https://github.com/user-attachments/assets/a4675750-7c93-4d7e-8b2f-2b66c0693077" />

## Results:


•Achieved ~98% accuracy on the test dataset

•Strong generalization across all 36 consonant classes

•Stable training with minimal overfitting

<br>
<br>
<br>

## [Github](https://github.com/Vishal-7003/Hindi-Consonant-CNN)

## [Complete Code Here](https://github.com/Vishal-7003/Hindi-Consonant-CNN/blob/main/Hindi_Consonant_Classification_CNN.ipynb)






