import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ==== 1. Paths ====
# base folder = MiniProject (yahi par yeh file hai)
base_dir = "."                 # current folder
img_size = (224, 224)
batch_size = 32

train_dir = os.path.join(base_dir, "train")
val_dir   = os.path.join(base_dir, "test")

# ==== 2. Load train & test data ====
train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

val_ds = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

print("Classes:", train_ds.class_names)   # yahan ['fresh', 'rotten'] aana chahiye

# ==== 3. Normalize ====
normalization = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization(x), y))

# ==== 4. Model ====
model = keras.Sequential([
    layers.Conv2D(16, 3, activation="relu", input_shape=img_size + (3,)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax")   # 2 classes: fresh, rotten
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==== 5. Train ====
model.fit(train_ds, validation_data=val_ds, epochs=5)

# ==== 6. Save model ====
model.save("freshness_model.h5")
print("✅ Model saved as freshness_model.h5")