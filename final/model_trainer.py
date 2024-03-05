import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Load the dataset into a DataFrame
df = pd.read_csv('dataset.csv')

# Drop non-numeric and irrelevant columns
X = df.drop(columns=['Chunk Name', 'Audio File', 'Block', 'Prolongation', 'Repetition', 'No of Words'])  # Exclude non-numeric and irrelevant columns
y_prolongation = df['Prolongation']
y_repetition = df['Repetition']
y_block = df['Block']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape features for CNN layer
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train_prolongation, y_temp_prolongation = train_test_split(X_reshaped, y_prolongation, test_size=0.3, random_state=42)
X_val, X_test, y_val_prolongation, y_test_prolongation = train_test_split(X_temp, y_temp_prolongation, test_size=0.5, random_state=42)

# Define a function to create and train model
def train_model(X_train, y_train, X_val, y_val, model_name):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Define a callback to save the best model during training
    checkpoint = ModelCheckpoint(f'{model_name}.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])
    
    return model

# Train and save prolongation model
prolongation_model = train_model(X_train, y_train_prolongation, X_val, y_val_prolongation, 'prolongation_model')

# Train and save repetition model
X_train, X_temp, y_train_repetition, y_temp_repetition = train_test_split(X_reshaped, y_repetition, test_size=0.3, random_state=42)
X_val, X_test, y_val_repetition, y_test_repetition = train_test_split(X_temp, y_temp_repetition, test_size=0.5, random_state=42)
repetition_model = train_model(X_train, y_train_repetition, X_val, y_val_repetition, 'repetition_model')

# Train and save block model
X_train, X_temp, y_train_block, y_temp_block = train_test_split(X_reshaped, y_block, test_size=0.3, random_state=42)
X_val, X_test, y_val_block, y_test_block = train_test_split(X_temp, y_temp_block, test_size=0.5, random_state=42)
block_model = train_model(X_train, y_train_block, X_val, y_val_block, 'block_model')
