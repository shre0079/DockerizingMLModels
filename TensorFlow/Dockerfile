# 1. Use official TensorFlow CPU image
FROM tensorflow/tensorflow:2.15.0

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy requirements first (better caching)
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy rest of the code
COPY . .

# 6. Run the training script
CMD ["python", "train.py"]
