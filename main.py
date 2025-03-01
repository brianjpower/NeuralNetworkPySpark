from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, hour, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.linalg import DenseVector
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
import numpy as np
from pyspark.sql import Row
from pyspark.ml.linalg import DenseVector


# Step 1: Initialize PySpark Session
spark = SparkSession.builder \
    .appName("EnergyConsumptionPySpark-TensorFlow") \
    .getOrCreate()

# Step 2: Load the Dataset
# Load the data as a PySpark DataFrame
df = spark.read.csv("data_energy_consumption.csv", header=True, inferSchema=True)

# Step 3: Preprocessing with PySpark
# Convert the 'date' column to timestamp and extract month/hour
df = df.withColumn("date", col("date").cast("timestamp"))
df = df.withColumn("month", month(col("date"))).withColumn("hour", hour(col("date")))

# Encode the 'Appliances' column into binary classification (0 = "normal", 1 = "high")
df = df.withColumn("label", when(col("Appliances") == "normal", 0).otherwise(1))

# Drop unnecessary columns
df = df.drop("date", "Appliances")

# Assemble the feature columns into a single feature vector
feature_cols = [col for col in df.columns if col != "label"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Scale the features using StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# Step 4: Split the Data into Train and Test Sets
train, test = df.randomSplit([0.7, 0.3], seed=42)


# Step 5: Convert PySpark DataFrames to TensorFlow-Compatible NumPy Arrays
# Define a function to convert Spark DataFrame to features (X) and labels (y)
def spark_df_to_numpy(df, features_col, label_col):
    df = df.select(features_col, label_col).rdd.map(lambda row: (row[0], row[1]))
    X, y = zip(*df.collect())
    X = np.array([x.toArray() for x in X])  # Convert DenseVector to numpy array
    y = np.array(y)  # Labels as numpy array
    return X, y

def spark_df_to_numpy_distributed(df, features_col, label_col):
    # Convert each row to NumPy-compatible tuple
    numpy_rdd = df.select(features_col, label_col).rdd.map(
        lambda row: (
            np.array(row[features_col].toArray()),  # Features
            row[label_col]  # Label
        )
    )
    # Collect as a list of NumPy arrays and split them into X and y
    numpy_data = numpy_rdd.collect()
    X, y = zip(*numpy_data)  # Unpack tuples
    return np.array(X), np.array(y)  # Convert list to final array


# Convert train and test sets to NumPy arrays
X_train, y_train = spark_df_to_numpy_distributed(train, "scaled_features", "label")
X_test, y_test = spark_df_to_numpy_distributed(test, "scaled_features", "label")

# Convert labels to one-hot encoding for TensorFlow
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Step 6: Build and Train the Neural Network with TensorFlow
# Get feature count and number of classes
input_shape = X_train.shape[1]
num_classes = y_train.shape[1]

# Define the neural network
model = Sequential([
    Input(shape=(input_shape,)),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Step 7: Evaluate the Neural Network
# Evaluate accuracy on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Step 8: Save the Model
model.save("energy_consumption_model.h5")

# Step 9: Predictions on New Data
# New data should go through similar preprocessing
new_data = spark.read.csv("new_energy_consumption_data.csv", header=True, inferSchema=True)
new_data = new_data.withColumn("date", col("date").cast("timestamp"))
new_data = new_data.withColumn("month", month(col("date"))).withColumn("hour", hour(col("date")))
new_data = new_data.drop("date", "Appliances")
new_data = assembler.transform(new_data)
new_data = scaler_model.transform(new_data)

# Convert new data to NumPy for predictions
X_new, _ = spark_df_to_numpy(new_data, "scaled_features", "label")

# Predict with the trained model
predictions = model.predict(X_new)
predicted_classes = np.argmax(predictions, axis=1)
print(f"Predicted Classes: {predicted_classes}")
