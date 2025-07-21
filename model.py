import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = 128
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory('data', target_size=(img_size, img_size),
                                         batch_size=32, subset='training', class_mode='binary')
val_data = datagen.flow_from_directory('data', target_size=(img_size, img_size),
                                       batch_size=32, subset='validation', class_mode='binary')


from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary: 1=focused, 0=distracted
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(train_data, validation_data=val_data, epochs=10)

model.save('focus_detection_model.h5')


from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the saved model
model = load_model('focus_detection_model.h5')

# Set image path
img_path = 'data/focused/focused_1750303008.jpg'  # Put your actual image file here
img_size = 128

# Load and preprocess the image
image = cv2.imread(img_path)
image = cv2.resize(image, (img_size, img_size))
image = image / 255.0
image = np.expand_dims(image, axis=0)

# Make prediction
pred = model.predict(image)[0][0]
label = "Focused" if pred > 0.5 else "Distracted"

# Show the result
output_img = cv2.imread(img_path)
cv2.putText(output_img, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 255, 0) if label == "Focused" else (0, 0, 255), 2)
cv2.imshow("Result", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
