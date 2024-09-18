from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import time
# ================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras import layers, models, optimizers
# ================================================================
import matplotlib.pyplot as plt
import numpy as np


# ############# DataLoader Class #############
class DataLoader:
    def __init__(self):
        print("Preparing data ... ... ... ")
        self.X_train = None ; self.X_test = None ;         self.y_train = None ; self.y_test = None
        # Load the Iris dataset
        iris = datasets.load_iris()  # contains data, target, frame, target_names
        X = iris.data  # each is an array of four features (sepal/petal length and width)
        y = iris.target  # label, 0,1,2

        # One-hot encode the labels
        encoder = OneHotEncoder()  # used to convert categorical labels into a binary (one-hot) encoded format.
        """
                Class 0 → [1, 0, 0]
                Class 1 → [0, 1, 0]
                Class 2 → [0, 0, 1]
        """
        y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        """
                train_test_split(X, y): This function splits the dataset into two parts: a training set and a testing set.
                X is the feature matrix (input data).
                y is the target labels (output data).
                random_state -> controls the randomness involved in the operation, ensuring that the results are reproducible.
        """

        # Standardize the feature values
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        print("Data loaded and prepared.")


    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

# ############# NeuralNetworkExperiment Class #############
class NeuralNetworkExperiment:
    def __init__(self, X_train, X_test, y_train, y_test, learning_rate):
        # Initialize with prepared data and learning rate
        self.X_train = X_train ; self.X_test = X_test ; self.y_train = y_train ; self.y_test = y_test ; self.learning_rate = learning_rate ;
        self.model = None

    # ############# Section 2: Build the Model #############
    def build_model(self):
        # Initialize the Sequential model
        self.model = models.Sequential()

        # Add the Input layer explicitly
        self.model.add(layers.Input(shape=(self.X_train.shape[1],)))

        # Add a 16-neuron hidden dense layer with the tanh activation function
        self.model.add(layers.Dense(16, activation='tanh'))

        # Add a 3-neuron output dense layer with the softmax activation function
        self.model.add(layers.Dense(3, activation='softmax'))

        # Compile the model with SGD optimizer, categorical cross-entropy loss, and accuracy metrics
        optimizer = optimizers.SGD(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # ############# Section 3: Train the Model #############
    def train_model(self):
        # Train the model with the specified parameters
        history = self.model.fit(self.X_train, self.y_train, epochs=40, batch_size=32, validation_split=0.1)
        """
        The model is trained using the fit method with:
        - epochs=40 -> Number of times the training data will be passed through the model.
        - batch_size=32 -> Number of samples per gradient update.
        - validation_split=0.1 -> Reserve 10% of training data for validation to check how well the model generalizes.
        """
        return history.history['accuracy']  # Return training accuracy for each epoch

def exercise_1_a (X_train, X_test, y_train, y_test):
    learning_rates = [0.003, 0.03, 0.06, 0.2, 0.5]  # Different learning rates to test
    epochs = 40
    runs_per_learning_rate = 20  # Number of times to run each experiment

    # Dictionary to store the average accuracy for each learning rate
    avg_accuracy_per_lr = {lr: np.zeros(epochs) for lr in learning_rates}
    print(avg_accuracy_per_lr)

    # Loop through each learning rate and run the experiment 20 times
    for lr in learning_rates:
        print(f"Running experiments for learning rate: {lr}")
        accuracies = []

        for run in range(runs_per_learning_rate):
            print(f"Run {run + 1} for learning rate {lr}")
            experiment = NeuralNetworkExperiment(X_train, X_test, y_train, y_test, learning_rate=lr)
            experiment.build_model()
            accuracy = experiment.train_model()
            accuracies.append(accuracy)

        # Compute the average accuracy over the 20 runs
        avg_accuracy = np.mean(accuracies, axis=0)
        avg_accuracy_per_lr[lr] = avg_accuracy

    # ############# Plotting the Results #############
    plt.figure(figsize=(10, 6))

    # Plot a curve for each learning rate
    for lr in learning_rates:
        plt.plot(range(1, epochs + 1), avg_accuracy_per_lr[lr], label=f'LR = {lr}')

    # Configure the plot
    plt.title('Training Accuracy vs Epochs for Different Learning Rates')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.legend(title="Learning Rate")
    plt.grid(True)

    plt.savefig('results_ex1_a.png', dpi=300, bbox_inches='tight')
    plt.show()


def exercise_1_b(X_train, X_test, y_train, y_test):
    # Set constant learning rate
    learning_rate = 0.02
    # Momentum values to test
    momentum_values = [0.0, 0.3, 0.6, 0.9, 0.99]
    epochs = 40
    runs_per_momentum = 20  # Number of times to run each experiment

    # Dictionary to store the average accuracy for each momentum value
    avg_accuracy_per_momentum = {momentum: np.zeros(epochs) for momentum in momentum_values}

    # Loop through each momentum value and run the experiment 20 times
    for momentum in momentum_values:
        print(f"Running experiments for momentum: {momentum}")
        accuracies = []

        for run in range(runs_per_momentum):
            print(f"Run {run + 1} for momentum {momentum}")
            # Build and compile the model with the current momentum value
            model = models.Sequential()
            model.add(layers.Dense(16, activation='tanh', input_shape=(X_train.shape[1],)))
            model.add(layers.Dense(3, activation='softmax'))

            # Compile model with SGD optimizer, constant learning rate, and varying momentum
            optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            # Train the model and record the training accuracy for each epoch
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)
            accuracies.append(history.history['accuracy'])

        # Compute the average accuracy over the 20 runs for each epoch
        avg_accuracy = np.mean(accuracies, axis=0)
        avg_accuracy_per_momentum[momentum] = avg_accuracy

    # ############# Plotting the Results #############
    plt.figure(figsize=(10, 6))

    # Plot a curve for each momentum value
    for momentum in momentum_values:
        plt.plot(range(1, epochs + 1), avg_accuracy_per_momentum[momentum], label=f'Momentum = {momentum}')

    # Configure the plot
    plt.title('Training Accuracy vs Epochs for Different Momentum Values')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.legend(title="Momentum")
    plt.grid(True)

    plt.savefig('results_ex1_b.png', dpi=300, bbox_inches='tight')
    plt.show()


# ############# Main Execution #############
if __name__ == "__main__":
    # Load the data once using DataLoader
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.get_data()

    # Method: each exercise will create its own DNNs
    # exercise_1_a(X_train, X_test, y_train, y_test)
    exercise_1_b(X_train, X_test, y_train, y_test)
