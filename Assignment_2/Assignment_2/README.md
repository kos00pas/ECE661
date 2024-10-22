# CNN Model Architecture for Fashion MNIST Classification
# Exercise 1
## Model Overview:

This model is designed for the Fashion MNIST dataset using Convolutional Neural Networks (CNN) with various improvements to enhance accuracy and prevent overfitting.

---

## Table: Model Decisions and Explanations

| **What**             | **Decision**              | **Reason**                                                                                                                                                                                                                           |
|----------------------|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Optimizer**         | Adam                      | 1. Less hyperparameters to tune: <br> &nbsp;&nbsp;&nbsp;&nbsp;a. adaptive learning rate <br> &nbsp;&nbsp;&nbsp;&nbsp;b. Inherently included <br> 2. Less worried about vanishing gradient: <br> &nbsp;&nbsp;&nbsp;&nbsp;a. adaptive learning rate <br> &nbsp;&nbsp;&nbsp;&nbsp;b. Inherently included <br> 3. Faster convergence <br> 4. Address risk of overfitting: <br> &nbsp;&nbsp;&nbsp;&nbsp;a. Dropout <br> &nbsp;&nbsp;&nbsp;&nbsp;b. L2 regularization <br> &nbsp;&nbsp;&nbsp;&nbsp;c. Batch normalization <br> &nbsp;&nbsp;&nbsp;&nbsp;d. Early stopping  |
| **Epochs**            | Early stopping            | 1. Prevent overfitting <br> 2. Save time by testing fewer epochs                                                                                                                                                                     |
| **Metrics**           | Accuracy + Validation Loss| Real-time monitoring with easy interpretation and ensure generalization by tracking validation loss alongside accuracy.                                                                                                               |
| **Learning Rate**     | Learning rate scheduler   | A learning rate scheduler was added to reduce the learning rate every 5 epochs to help the model converge more efficiently and avoid overfitting by slowly adjusting the learning rate.                                                 |
| **Data Augmentation** | Rotation: 15°, Shifts: 0.1, Flip | Preprocessing technique to introduce variations into the dataset, preventing the model from memorizing the training data and improving generalization. Rotations, shifts, and horizontal flips were applied during training.            |
| **Loss function**     | Categorical cross entropy | Widely used for multi-classification tasks → standard choice                                                                                                                                                                         |
| **Convolution**       | 3x3 Convolutions + Dilated Convolutions | 1. **3x3 filters** effectively capture local features (edges, textures) while keeping the number of parameters reasonable. <br> 2. **Dilated convolutions** expand the receptive field without increasing the number of parameters, helping to capture larger spatial context. <br> 3. **Batch normalization** added after each convolution layer to stabilize training and reduce sensitivity to initialization. <br> 4. **Max pooling** (2x2) to reduce spatial dimensions while retaining key features. |
| **Dropout**           | In Fully Connected Layer (0.5) | Prevent overfitting by deactivating 50% of neurons in fully connected layers. A 0.5 dropout rate is commonly used to add regularization in deeper layers.                                                                             |
| **Pooling - subsampling** | Max Pooling (2x2)       | **2x2 Max Pooling** reduces the spatial dimensions of feature maps, allowing the network to focus on the most important features while reducing computation. It also helps make the model more invariant to small translations in the image. |
| **Batch normalization** | After Conv Layers        | Stabilizes training, reduces sensitivity to initialization, prevents vanishing gradients, and standardizes activations.                                                                                                              |
| **L2 regulation**     | On Fully Connected Layers | Reduces overfitting by penalizing large weights and simplifying complexity.                                                                                                                                                          |
| **Flatten**           | Prepares data to feature vector | Converts 2D convolutional output to 1D for the fully connected layers.                                                                                                                                                                |
| **Fully connected**   | 128 neurons (first), 64 or 32 neurons (optional second) | Final decision-making: maps flattened feature vector to output, 128 neurons for high capacity, fewer neurons in subsequent layers for refinement.                                                                                      |
| **Activation Function** | ReLU                     | **ReLU** is used in convolutional and fully connected layers to address the vanishing gradient problem and introduce non-linearity, allowing the model to learn more complex patterns.                                                 |
| **Output layer**      | Softmax activation (10 neurons) | Probabilistic interpretation with clear decision-making to distinguish between classes.                                                                                                                                               |
| **Batch size**        | 64                        | A batch size of 64 is commonly used as it strikes a good balance between training speed and generalization performance. It’s small enough to provide good gradient updates and large enough for computational efficiency.               |
| **Global Average Pooling** | After final convolutional layer | Replaces fully connected layers by averaging the spatial dimensions of feature maps, reducing overfitting and making the model lighter. **GAP** simplifies the architecture while retaining important features.                         |
| **SE Block (Squeeze and Excitation)** | Added after each convolutional block | Recalibrates channel-wise feature responses by giving more importance to the most relevant channels. **SE blocks** improve the model's ability to focus on important features, enhancing accuracy without adding much computational cost. |

---

## Model Architecture (Layer Order):

1. **Input Layer**:
   - Input size: **28x28 grayscale image**.

2. **Convolutional Block 1**:
   - **Conv Layer 1**: 32 filters, **3x3** convolution, **ReLU** activation.
   - **Batch Normalization**: After Conv Layer 1.
   - **SE Block**: Squeeze and Excitation block applied to recalibrate feature maps.
   - **Max Pooling**: **2x2** pooling.

3. **Convolutional Block 2**:
   - **Conv Layer 2**: 64 filters, **3x3** convolution, **ReLU** activation.
   - **Batch Normalization**: After Conv Layer 2.
   - **SE Block**: Squeeze and Excitation block applied again.
   - **Max Pooling**: **2x2** pooling.

4. **Convolutional Block 3** (Optional, based on model complexity):
   - **Conv Layer 3**: 128 filters, **Dilated convolution (3x3, dilation rate 2)**, **ReLU** activation (for capturing larger spatial context).
   - **Batch Normalization**: After Conv Layer 3.
   - **SE Block**: Squeeze and Excitation block applied again.
   - **Max Pooling**: **2x2** pooling.

5. **Global Average Pooling**:
   - Replaces the traditional fully connected layers by averaging over the entire spatial dimension, producing a 1D vector.

6. **Fully Connected Block 1** (Optional based on GAP use):
   - **Fully Connected Layer 1**: 128 neurons, **ReLU** activation.
   - **Dropout**: **0.5** dropout to reduce overfitting.

7. **Output Layer**:
   - **Output Layer**: 10 neurons (for the 10 classes), **Softmax** activation for multi-class classification.

This final architecture balances **complexity and performance** with efficient feature extraction, improved focus on important features, and regularization through GAP and dropout.

---
## Discussion 
The model was trained on the Fashion MNIST dataset using 20 epochs with early stopping and data augmentation techniques. It achieved a final test accuracy of 90.94% with a validation accuracy of 90.98% at its peak. A learning rate scheduler was applied to gradually reduce the learning rate, which improved convergence and minimized overfitting. The use of Squeeze-and-Excitation (SE) blocks, global average pooling, and data augmentation (including random rotations, shifts, and flips) contributed to the model's strong generalization ability. The model maintained consistent performance across training, validation, and test sets.
### Improvements to Prevent Overfitting:

- **Data Augmentation**: Random rotations (up to 15°), width and height shifts (10% of the image), and horizontal flips were applied to introduce variability into the dataset and improve generalization.
- **Learning Rate Scheduler**: The learning rate was reduced by half every 5 epochs to allow for more refined weight updates as the model converges.

---
# Exercise 2

## Concept of Residual Connections

Residual connections are direct connections that skip one or more layers. The idea is that instead of learning the transformation $H(x)$, the model learns the residual $F(x) = H(x) - x$. This makes it easier for the model to optimize because it can simply "add" the residual to the input. If the residual is zero, the layers act as identity mappings, allowing the model to learn deeper networks without degradation in performance.

Residual connections help mitigate the vanishing gradient problem by ensuring that gradients are passed through the shortcut paths, allowing for deeper networks that can still be trained effectively. This technique was first popularized in the ResNet architecture, which demonstrated that extremely deep networks could be trained with good performance by utilizing these shortcuts.

## Modifications for Exercise 2

For this task, we modified the forward function of the CNN model created in Exercise 1 to incorporate residual connections. Since CNNs typically have a pyramid structure where the feature map dimensions are progressively reduced (due to pooling or stride convolutions), the residual connections require adjustments in dimensions. Specifically, we used convolutional layers in the residual paths to match the dimensions of the input and output feature maps. This ensures that the feature maps added via the residual connections are compatible with the main network's dimensions. If needed, zero-padding was used to match dimensions when the number of channels or spatial dimensions differed.

## Results and Comparison

After implementing residual connections, the model was trained on the Fashion MNIST dataset. The **test accuracy** for this model was **92.62%**, an improvement over the **90.94%** achieved in Exercise 1. Similarly, the **validation accuracy** reached **92.47%**, higher than the **90.98%** from Exercise 1. This demonstrates that the introduction of residual connections led to better performance, both in terms of accuracy and generalization.

Moreover, the **validation loss** in Exercise 2 was **0.2117**, a significant reduction compared to **0.2504** in Exercise 1. This confirms that the residual network was more effective in reducing overfitting and ensuring smoother convergence of the loss function.
[ask2_.py](ask2_.py)
[README.md](README.md)