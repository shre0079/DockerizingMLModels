import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tf_model.keras")

# 1. Create dummy dataset
X = np.random.rand(1000, 20)
y = np.random.randint(0, 2, (1000, 1))

# 2. Train-Test split (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Build model
model = Sequential([
    Dense(32, activation='relu', input_shape=(20,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 4. Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. Train
print("Training started...")
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 6. Test / Evaluate
print("\nTesting on unseen data...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 7. Save model
model.save(MODEL_PATH)
print("Model saved at:", MODEL_PATH)

# 8. Load model
loaded_model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

# 9. Prediction
sample_input = np.random.rand(1, 20)
prediction = loaded_model.predict(sample_input)

print("Prediction probability:", prediction)
print("Predicted class:", int(prediction > 0.5))