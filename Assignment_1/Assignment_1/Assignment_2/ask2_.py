import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.layers import GlobalAveragePooling2D



def load_fashion_mnist_data():
    print("Start Loading")
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print("Done Loading")

    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]
    num_classes = 10
    train_labels = to_categorical(train_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

    return train_images, train_labels, test_images, test_labels

def se_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    return layers.multiply([input_tensor, se])

def build_model_exercise_1(input_shape, num_classes=10):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = se_block(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def lr_schedule(epoch, lr):
    if epoch % 5 == 0 and epoch > 0:
        return lr * 0.5
    return lr

def train_model(model, train_images, train_labels, test_images, test_labels, batch_size=64, epochs=20):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(train_images)

    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    # Display the model summary
    model.summary()

    # Train the model
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(test_images, test_labels),
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                            lr_scheduler
                        ])

    return history
def residual_block(x, filters, kernel_size=3, padding='same', strides=1):
    """Defines a Residual Block with two convolutional layers and a skip connection."""
    shortcut = x

    # First convolution
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Second convolution
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, activation=None)(x)
    x = layers.BatchNormalization()(x)

    # If the dimensions don't match, use a 1x1 convolution to match them
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = layers.Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(shortcut)

    # Add the shortcut (skip connection)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    return x
def build_model_exercise_2(input_shape, num_classes=10):
    """Builds the Residual Network model for Exercise 2."""
    inputs = layers.Input(shape=input_shape)

    # First Conv Layer without Residual
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # First Residual Block
    x = residual_block(x, filters=64)

    # Second Residual Block
    x = residual_block(x, filters=128, strides=2)

    # Third Residual Block
    x = residual_block(x, filters=128)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output Layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Build the model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def exercise_1():
    train_images, train_labels, test_images, test_labels = load_fashion_mnist_data()
    model = build_model_exercise_1(input_shape=(28, 28, 1), num_classes=10)
    history = train_model(model, train_images, train_labels, test_images, test_labels)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}")
    return model, history
def exercise_2():
    train_images, train_labels, test_images, test_labels = load_fashion_mnist_data()
    model = build_model_exercise_2(input_shape=(28, 28, 1), num_classes=10)
    history = train_model(model, train_images, train_labels, test_images, test_labels)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}")
    return model, history

if __name__ == "__main__":
    #exercise_1()
    exercise_2()
