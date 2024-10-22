from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load Reuters dataset, only considering the top 10,000 most frequent words
max_features = 10000  # number of words to consider as features
maxlen = 20  # cut texts after this number of words
batch_size = 32

# Load the data (as integer sequences)
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)

# Pad sequences (to ensure all sequences have the same length)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Build the LSTM model
model = Sequential()
model.add(Embedding(max_features, 32))  # Embedding layer
model.add(LSTM(32))  # LSTM layer
model.add(Dense(46, activation='softmax'))  # Output layer for multi-class classification

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model on test data
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
