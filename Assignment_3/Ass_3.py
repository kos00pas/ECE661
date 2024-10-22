from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load the dataset
# Only considering the top 10,000 most frequently occurring words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Pad the data so that all sequences have the same length
x_train = sequence.pad_sequences(x_train, maxlen=500)
x_test = sequence.pad_sequences(x_test, maxlen=500)
print('Train set shape:', x_train.shape)
print('Test set shape:', x_test.shape)
