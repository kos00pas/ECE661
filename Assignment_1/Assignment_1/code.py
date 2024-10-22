from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import time
# ================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras import layers, models, optimizers,regularizers
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
        self.X_train = X_train ; self.X_test = X_test ; self.y_train = y_train ; self.y_test = y_test ; self.learning_rate = learning_rate ; self.model = None

    # ############# Section 2: Build the Model #############
    def build_model(self, momentum=None):
        # Initialize the Sequential model
        self.model = models.Sequential()

        # Add the Input layer explicitly
        self.model.add(layers.Input(shape=(self.X_train.shape[1],)))

        # Add a 16-neuron hidden dense layer with the tanh activation function
        self.model.add(layers.Dense(16, activation='tanh'))

        # Add a 3-neuron output dense layer with the softmax activation function
        self.model.add(layers.Dense(3, activation='softmax'))

        # Compile the model with SGD optimizer
        optimizer = optimizers.SGD(learning_rate=self.learning_rate,
                                   momentum=momentum) if momentum is not None else optimizers.SGD(
                                    learning_rate=self.learning_rate)

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
    # Different learning rates to test
    learning_rates = [0.003, 0.03, 0.06, 0.2, 0.5]
    epochs = 40
    runs_per_learning_rate = 20  # Number of times to run each experiment

    avg_accuracy_per_lr = {lr: np.zeros(epochs) for lr in learning_rates}
    # Setup Dictionary to store the average accuracy for each learning rate

    """#######################################################################"""
    for lr in learning_rates:
        # Loop through each learning rate and run the experiment 20 times
        print(f"Running experiments for learning rate: {lr}")
        accuracies = []

        for run in range(runs_per_learning_rate):
            print(f"Run {run + 1} for learning rate {lr}")
            """#############"""
            experiment = NeuralNetworkExperiment(X_train, X_test, y_train, y_test, learning_rate=lr)
            experiment.build_model()
            accuracy = experiment.train_model()
            accuracies.append(accuracy)
            """#############"""

        # Compute the average accuracy over the 20 runs
        avg_accuracy = np.mean(accuracies, axis=0)
        avg_accuracy_per_lr[lr] = avg_accuracy
    """#######################################################################"""

    # ############# Plotting the Results #############
    plt.figure(figsize=(10, 6))

    # Plot a curve for each learning rate
    for lr in learning_rates:
        plt.plot(range(1, epochs + 1), avg_accuracy_per_lr[lr], label=f'LR = {lr}')
        #plt.plot(x, y, format_string='', **kwargs)


    # Configure the plot
    plt.title('Training Accuracy vs Epochs for Different Learning Rates')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.legend(title="Learning Rate")
    plt.grid(True)
    plt.savefig('./images/results_ex1_a.png', dpi=300, bbox_inches='tight')
    plt.show()


def exercise_1_b(X_train, X_test, y_train, y_test):
    learning_rate = 0.02
    momentum_values = [0.0, 0.3, 0.6, 0.9, 0.99]
    epochs = 40
    runs_per_momentum = 20  # Number of times to run each experiment

    avg_accuracy_per_momentum = {momentum: np.zeros(epochs) for momentum in momentum_values}
    print(avg_accuracy_per_momentum)
    for momentum in momentum_values:
        print(f"Running experiments for momentum: {momentum}")
        accuracies = []

        for run in range(runs_per_momentum):
            print(f"Run {run + 1} for momentum {momentum}")
            """#############"""
            experiment = NeuralNetworkExperiment(X_train, X_test, y_train, y_test, learning_rate)
            experiment.build_model(momentum=momentum)  # Modify build_model to accept momentum
            accuracy = experiment.train_model()
            accuracies.append(accuracy)
            """#############"""

        # Compute the average accuracy over the 20 runs
        avg_accuracy = np.mean(accuracies, axis=0)
        avg_accuracy_per_momentum[momentum] = avg_accuracy

    # Plotting the results
    plt.figure(figsize=(10, 6))
    for momentum in momentum_values:
        plt.plot(range(1, epochs + 1), avg_accuracy_per_momentum[momentum], label=f'Momentum = {momentum}')

    plt.title('Training Accuracy vs Epochs for Different Momentum Values')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.legend(title="Momentum")
    plt.grid(True)
    plt.savefig('./images/results_ex1_b.png', dpi=300, bbox_inches='tight')
    plt.show()

def exercise_1_c(X_train, X_test, y_train, y_test ):
    neuron_list = [4, 8, 16, 32, 64]
    epochs = 40
    batch_size = 32
    results = {}

    for neurons in neuron_list:
        accuracies = []
        for i in range(20):
            """#############"""
            """Build a New NN"""
            model = models.Sequential()
            model.add(layers.Input(shape=(X_train.shape[1],)))
            model.add(layers.Dense(neurons, activation='tanh')) # Neurons
            model.add(layers.Dense(y_train.shape[1], activation='softmax'))
            """#############"""
            optimizer = optimizers.SGD(learning_rate=0.02, momentum=0.9)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            """#############"""
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
            accuracies.append(history.history['accuracy'])
            """#############"""

        # Average the accuracy over 20 runs for each neuron setting
        avg_accuracy = np.mean(accuracies, axis=0)
        results[neurons] = avg_accuracy

    # Plotting the average training accuracy for each number of neurons
    plt.figure(figsize=(10, 6))
    for neurons, accuracy in results.items():
        plt.plot(accuracy, label=f'{neurons} Neurons')

    plt.title('Training Accuracy vs Epochs for Different Hidden Layer Neuron Numbers')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.legend()
    plt.savefig('./images/results_ex1_c.png', dpi=300, bbox_inches='tight')
    plt.show()


def exercise_1_d(X_train, X_test, y_train, y_test):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD
    import numpy as np
    import matplotlib.pyplot as plt

    # Define activation functions to test
    activations = ['tanh', 'sigmoid', 'relu']

    # Training parameters
    learning_rate = 0.02
    momentum = 0.9
    epochs = 40
    batch_size = 32
    n_hidden_neurons = 16
    n_classes = y_train.shape[1]

    # Store the results for each activation function
    results = {activation: np.zeros(epochs) for activation in activations}
    """#######################################################################"""
    for activation in activations:
        for _ in range(20):
            """#############"""
            model = Sequential()
            model.add(Dense(n_hidden_neurons, input_dim=X_train.shape[1], activation=activation))
            model.add(Dense(n_classes, activation='softmax'))
            """#############"""
            optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            """#############"""
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1)
            results[activation] += np.array(history.history['accuracy'])
        """#############"""
        results[activation] /= 20
    """#######################################################################"""

    # Plot the results
    plt.figure()
    for activation in results:
        plt.plot(np.arange(1, epochs + 1), results[activation], label=activation)

    plt.title('Training Accuracy for Different Activation Functions')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./images/results_ex1_d.png', dpi=300, bbox_inches='tight')

    plt.show()

    # Print out the best performing activation function
    best_activation = max(results, key=lambda x: results[x][-1])
    print(f'The best performing activation function is: {best_activation}')



def exercise_1_e(X_train, X_test, y_train, y_test):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.metrics import Accuracy

    batch_sizes = [1, 8, 16, 32, 64]
    epochs = 40
    learning_rate = 0.02
    momentum = 0.9
    hidden_neurons = 16
    activation_function = 'tanh'

    # Store average accuracy for each batch size
    batch_size_accuracy = {}

    """#######################################################################"""
    for batch_size in batch_sizes:
        print(f"Training with batch size: {batch_size}")
        accuracies = []
        for _ in range(20):
            """#############"""
            model = Sequential()
            model.add(Dense(hidden_neurons, activation=activation_function, input_shape=(X_train.shape[1],)))
            model.add(Dense(3, activation='softmax'))  # 3 output classes for Iris dataset
            """#############"""
            optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
            model.compile(optimizer=optimizer,
                          loss=CategoricalCrossentropy(),
                          metrics=['accuracy'])
            """#############"""
            history = model.fit(X_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0.1,
                                verbose=0)
            """#############"""
            accuracies.append(history.history['accuracy'])
        """#######################################################################"""

        avg_accuracy = np.mean(accuracies, axis=0)
        batch_size_accuracy[batch_size] = avg_accuracy

    # Plotting the results
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        plt.plot(range(1, epochs + 1), batch_size_accuracy[batch_size], label=f'Batch Size {batch_size}')

    plt.title('Training Accuracy vs Epochs for Different Batch Sizes')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/results_ex1_e.png', dpi=300, bbox_inches='tight')

    plt.show()


def exercise_2(X_train, X_test, y_train, y_test):
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.losses import CategoricalCrossentropy
    import os
    import pandas as pd

    # Define the combinations of parameters
    learning_rates = [0.2, 0.5]
    neurons_list = [32, 64]
    batch_sizes = [32, 64]
    epochs = 50

    # Dictionary to hold the accuracy results for each combination
    results = []
    best_accuracy = 0
    best_params = {}

    # Ensure the ./images directory exists
    os.makedirs('./images', exist_ok=True)
    """#######################################################################"""
    for lr in learning_rates:
        for neurons in neurons_list:
            for batch_size in batch_sizes:
                print(f"Running model with lr={lr}, neurons={neurons}, batch_size={batch_size}")
                """#############"""
                model = Sequential()
                model.add(Dense(neurons, input_shape=(X_train.shape[1],), activation='relu'))
                model.add(Dense(3, activation='softmax'))  # 3 output classes for Iris dataset
                """#############"""
                optimizer = SGD(learning_rate=lr, momentum=0.9)
                model.compile(optimizer=optimizer,
                              loss=CategoricalCrossentropy(),
                              metrics=['accuracy'])
                """#############"""
                history = model.fit(X_train, y_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_split=0.1,
                                    verbose=0)
                """#######################################################################"""

                # Save the plot
                plt.figure(figsize=(10, 6))
                plt.plot(history.history['accuracy'], label='Training Accuracy')
                plt.title(f'Accuracy vs Epochs (LR={lr}, Neurons={neurons}, Batch Size={batch_size})')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)

                # Save and show the plot
                plot_filename = f'lr_{lr}_neurons_{neurons}_batch_size_{batch_size}.png'
                plt.savefig(f'./images/{plot_filename}', dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()

                # Store accuracy results for the table (epochs 10, 15, 40)
                accuracy_epoch_10 = history.history['accuracy'][9]  # 10th epoch (index 9)
                accuracy_epoch_15 = history.history['accuracy'][14]  # 15th epoch (index 14)
                accuracy_epoch_40 = history.history['accuracy'][39]  # 40th epoch (index 39)

                # Find best accuracy between epochs 10, 15, and 40, and record the corresponding epoch
                best_model_accuracy = max(accuracy_epoch_10, accuracy_epoch_15, accuracy_epoch_40)
                if best_model_accuracy == accuracy_epoch_10:
                    best_model_epoch = 'epoch: 10'
                elif best_model_accuracy == accuracy_epoch_15:
                    best_model_epoch = 'epoch: 15'
                else:
                    best_model_epoch = 'epoch: 40'

                # Track the best model overall
                if best_model_accuracy > best_accuracy:
                    best_accuracy = best_model_accuracy
                    best_params = {'lr': lr, 'neurons': neurons, 'batch_size': batch_size}

                # Append the results (the best accuracy between epochs 10, 15, and 40)
                results.append({
                    'lr': lr,
                    'neurons': neurons,
                    'batch_size': batch_size,
                    'accuracy_epoch_10': accuracy_epoch_10,
                    'accuracy_epoch_15': accuracy_epoch_15,
                    'accuracy_epoch_40': accuracy_epoch_40,
                    'best_accuracy': f"{best_model_accuracy} ({best_model_epoch})"
                    # Best accuracy and corresponding epoch
                })

    # Convert results to a pandas DataFrame for the table
    results_df = pd.DataFrame(results)

    # Find the models with the best accuracy for epochs 10, 15, and 40 respectively
    best_at_epoch_10 = results_df.loc[results_df['accuracy_epoch_10'].idxmax()]
    best_at_epoch_15 = results_df.loc[results_df['accuracy_epoch_15'].idxmax()]
    best_at_epoch_40 = results_df.loc[results_df['accuracy_epoch_40'].idxmax()]

    # Find the best overall model
    best_overall_model = results_df.loc[results_df['best_accuracy'].apply(lambda x: float(x.split()[0])).idxmax()]

    # Create a new row for the best models at epochs 10, 15, and 40, and the overall best model
    best_combined_row = pd.DataFrame([{
        'lr': f"({best_at_epoch_10['lr']}, {best_at_epoch_15['lr']}, {best_at_epoch_40['lr']})",
        'neurons': f"({best_at_epoch_10['neurons']}, {best_at_epoch_15['neurons']}, {best_at_epoch_40['neurons']})",
        'batch_size': f"({best_at_epoch_10['batch_size']}, {best_at_epoch_15['batch_size']}, {best_at_epoch_40['batch_size']})",
        'accuracy_epoch_10': best_at_epoch_10['accuracy_epoch_10'],
        'accuracy_epoch_15': best_at_epoch_15['accuracy_epoch_15'],
        'accuracy_epoch_40': best_at_epoch_40['accuracy_epoch_40'],
        'best_accuracy': f"{best_overall_model['best_accuracy']} ({best_overall_model['lr']}, {best_overall_model['neurons']}, {best_overall_model['batch_size']})"
    }])

    # Use pd.concat to append the new row to the dataframe
    results_df = pd.concat([results_df, best_combined_row], ignore_index=True)

    # Print the table in terminal format
    print("\nResults Table:")
    print(results_df.to_string(index=False))

    # Save the results table as a CSV file
    results_df.to_csv('./images/results_table.csv', index=False)

    # Print the best model
    print(
        f"\nBest Model: LR={best_params['lr']}, Neurons={best_params['neurons']}, Batch Size={best_params['batch_size']}")
    print(f"Best Accuracy: {best_accuracy}")
    print("Experiment completed.")

    # Plot results for the accuracies
    def plot_results(df):
        plt.figure(figsize=(14, 8))

        # Plot lines for each epoch and the best accuracy
        plt.plot(df.index[:-1], df['accuracy_epoch_10'][:-1], marker='o', label='accuracy_epoch_10', color='b')
        plt.plot(df.index[:-1], df['accuracy_epoch_15'][:-1], marker='o', label='accuracy_epoch_15', color='orange')
        plt.plot(df.index[:-1], df['accuracy_epoch_40'][:-1], marker='o', label='accuracy_epoch_40', color='g')
        plt.plot(df.index[:-1], df['best_accuracy'][:-1].apply(lambda x: float(x.split()[0])), marker='o',
                 label='best_accuracy', color='r')

        # Set the y-axis limits from 0.9 to 1
        plt.ylim(0.9, 1)

        # Add labels and title
        plt.xlabel('Model Combinations (Index)')
        plt.ylabel('Accuracy')
        plt.title('Accuracies for Each Epoch and Best Accuracy (Y-axis from 0.9 to 1)')

        # Add ticks and xtick labels for the models
        plt.xticks(df.index[:-1], [f"LR={lr}, N={neurons}, BS={bs}" for lr, neurons, bs in
                                   zip(df['lr'][:-1], df['neurons'][:-1], df['batch_size'][:-1])], rotation=45)

        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.tight_layout()
        plt.savefig('./images/results_222_limited_yaxis.png', dpi=300, bbox_inches='tight')

        # Show the plot
        plt.show()

    # Call the function to plot the results
    plot_results(results_df)


# ############# Main Execution #############
if __name__ == "__main__":
    # Load the data once using DataLoader
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.get_data()

    """# Method: each exercise will create its own DNNs"""

    # exercise_1_a(X_train, X_test, y_train, y_test)
    # exercise_1_b(X_train, X_test, y_train, y_test)
    # exercise_1_c(X_train, X_test, y_train, y_test)
    # exercise_1_d(X_train, X_test, y_train, y_test)
    # exercise_1_e(X_train, X_test, y_train, y_test)
    # exercise_2(X_train, X_test, y_train, y_test)
