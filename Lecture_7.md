# Optimizing deep neural networks

Depth vs Error
: Shallow Networks:
   - Lower training and testing errors.
  -  Easier to optimize but struggle with complex patterns.
: Deep Networks:
-  Potential for higher error due to optimization difficulties (e.g., vanishing gradients).
-  Without proper construction, optimization becomes harder.

### Solution for Training Deeper Networks

#### 1. Start with a Shallow Network's Trained Layers (Knowledge Transfer)
- **Why?**
    - A pre-trained shallow network already "knows" basic patterns from the data.
    - Adding more layers builds on this knowledge, making training faster and more effective compared to starting from scratch.

#### 2. Add Extra Layers Initialized with Identity Mapping
- Initialize additional layers with the identity function: \( H(x) = x \).
- At the start, these layers pass inputs directly without altering them (identity mapping).
- **Why?**
    - The network initially behaves like the original shallow network, avoiding performance degradation.
    - During training, these extra layers gradually adapt to learn more complex transformations, enabling the network to capture intricate patterns.


Plain net
: Learns full mapping H(x) directly.
: Struggles with vanishing gradients and degradation in deeper architectures.



Deep residual learning
: Introduces identity skip connections:
: H(x)=F(x)+x, where F(x) is the residual.
: Solves optimization challenges , Eases gradient flow, reducing vanishing gradient issues.
: Enables effective training of very deep networks.


Functions 
: local , global max/min, saddle points
: Saddle points must be handled carefully because they have zero gradients like minima or maxima while there are no optimal solutions
:  they can trap optimization algorithms, slowing convergence or causing them to fail in high-dimensional spaces.


Derivatives of multivariate Functions 
: ![img.png](img/img.png)
: Hesian = second derivative 
: ![img_1.png](img/img_1.png)`


Convex
:  Easier to optimize since any local minimum is also the global minimum.

Non-Convex Functions:
: Contain multiple local minima, maxima, and saddle points.
: Harder to optimize since algorithms can get stuck in suboptimal points.
: Deep learning heavily deals with non-convex loss functions due to the high dimensionality and complex architectures.
: While global minima are not guaranteed, gradient-based methods like SGD often find good local minima or saddle points, which usually suffice for practical performance.
: ![img_2.png](img/img_2.png)


Optimization via Gradient Descent
: Iteratively adjusts parameters to minimize a loss function by moving in the direction of the negative gradient.
: Sensitive to learning rate.
: Can get stuck in saddle points or local minima.
: types of GD : batch , mini-batch , Stochastic 

learning rate 
: Too High - Overshoots the optimal solution, causes instability 
: Too low  - converges very slow and increases training time 
: solution : adaptive/ dynamically : Adam
: approaches for reduce learning rate decay : reduce lr by factor every few epochs or by constant 

![img_26.png](img/img_26.png)

why gd can not guarantee reach the global minimum? 
: Non-Convex Loss Functions: Gradient descent may get stuck in a local minimum or plateau at a saddle point
: The starting point determines the optimization path
: Gradient descent only considers local information
: In high-dimensional spaces, the probability of encountering saddle points or flat regions increases, making optimization unpredictable.


Vanishing gradient problem
: occurs when gradients become extremely small (0-1) during backpropagation in deep networks
: leads to  Very slow parameter updates  , Certain layers (especially early ones) learning almost nothing.
: making them effectively un-trainable
: Solutions: ReLu , Weight init , batch norm , ResNet
: not use sigmoid because has derivatives 0-0.25 and also cause saturation 


Leaky ReLU 
: allows a small, non-zero gradient for negative inputs:

Parametric ReLU (PReLU)
: Generalizes Leaky ReLU by making `α` learnable during training


Benefits Over Standard ReLU:
: Reduces risk of gradient starvation 
: Enhances representational power ( adapt α) 

  Classification vs Regression  Loss Function Goals
: Classification: Minimize the distance between predicted probabilities and true class probabilities.
: Regression: Minimize the error between predicted and actual continuous values.

Underfitting
:  model is too simple to capture the  patterns in the data.
: Results in poor performance on both training and test sets 

Underfitting causes 
: Model complexity 
: insufficient training 
: excessive regularization
: missing or irrelevant features 

Underfitting symptoms
: Both training and test errors are high.
: Little to no improvement in training loss over time.

Underfitting solutions 
: increase model complexity 
: train longer 
: use more meaningful input data
: lower L1/L2 penalty : because the shrinking or zeroing some weights 
: lr ensure it allows the model to effectively learn

L1/L2
: L1: Encourages sparsity by forcing some weights to zero.
: L2: Penalizes large weights, promoting smaller, smoother values.

Overfitting
: Results in high performance on training data but poor generalization to unseen test data.

Overfitting symptoms 
: Very low training error but high test error.
: Model performs well on training data but fails on new data.

Overfitting causes 
: too many neurons / layers /patters 
: insufficient training data to generalize effectively 
: lack of regularization 
: over-training 

Overfitting solutions
: regularization 
: increase training data , data augmentation
: reduce model complexity 
: monitor performance on validation ( decrease or stop while training increasing  ) 
: batch normalization 

![img_7.png](img/img_7.png)

Parameters vs hyper-parameters: 
: 

| **Aspect**           | **Parameters**                     | **Hyperparameters**                 |
|----------------------|------------------------------------|-------------------------------------|
| **Learned During Training** | Yes                               | No                                  |
| **Set By**           | Model training process            | User or external optimization       |
| **Examples**         | Weights, biases                   | Learning rate, number of layers     |
| **Optimization Algorithms** | Gradient-based methods like SGD, Adam, RMSprop | Not optimized directly during training |
| **Tuning**           | Adjusted via optimization algorithms (e.g., SGD, Adam) | Grid Search, Random Search, Bayesian Optimization, Hyperband |


Model Hyper-parameters
: Number of Layers
: Number of Neurons per Layer
: Activation Functions (e.g., ReLU, Sigmoid, Tanh, Leaky ReLU)

Training Hyperparameters
: Learning Rate 
: Batch Size
: Number of Epochs
: Momentum
: Weight Initialization (e.g., Xavier, He)


optimizing hyperparameters
: small-scale problems:  grid search or random search suffices.
: larger problems:  Bayesian optimization or Hyperband offers more efficiency.

methods for hyperparameter tuning
: grid search : Systematically explores a predefined set of hyperparameter values by evaluating all possible combinations.
: Random search :  Samples hyperparameter combinations randomly from specified distributions
: bayesian optimization  : Builds a probabilistic model of the objective function and uses it to select promising hyperparameter combinations to evaluate.



Data Preprocessing 
: Normalization/Standardization Method
: Augmentation Parameters (e.g., rotation, flipping, cropping


Iterative Hyperparameter Tuning Process:
: Start with an idea (e.g., number of layers, learning rate).
: Train the model on the training set.
: Validate the idea on the dev(val) set.
: Refine and repeat until performance is satisfactory.
: Evaluate the final model on the test set for unbiased results.


Bias 
: error due to overly simplistic assumptions in the model ( no patterns capturing ) 
: leads to underfitting  
: solution : increase model complexity 
: ***does not refer to the same bias term (b) in the perceptron equation

Variance
: error duo to the model being too sensitive to small fluctuations or noise in training data 
: leads to overfitting 
: solution : use regularization , reduce model complexity  , add more training data 

why high variance is no the right one? 
: it generalizes poorly to unseen data

![img_6.png](img/img_6.png)



Bias/variance trade-off
: good balance = generalization 
: Ideal: Low bias + Low variance = "Just right" model.
: Monitor the difference between training and validation errors:
- Both errors high = Underfitting (high bias).
- Large gap = Overfitting (high variance).

Regularization 
: techniques used to reduce the generalization error
:  discouraging the model from learning [noise] or [overly complex patterns] in the training data.

Reg technique:  parameter norm penalties
: adding penalty to obj function
: Reduces overfitting by discouraging large weights.
: L1 and L2 Regularization are two primary subcategories to specify Ω(θ)
: J=ℓ(θ)+λΩ(θ)

regularization techniques
: Model-based: L1/L2, Dropout, Spatial Dropout, Max-Norm, Weight Decay.
: Data-based: Data Augmentation, Mixup, Cutout.
: Process-based: Early Stopping, Batch Normalization, Label Smoothing.


L1/L2
: L1: Encourages sparsity by forcing some weights to zero.
: L2: Penalizes large weights, promoting smaller, smoother values.



L2 Regularization & L1 Regularization
: 

| **Aspect**           | **L1 Regularization (Lasso)**                          | **L2 Regularization (Ridge)**                          |
|----------------------|--------------------------------------------------------|--------------------------------------------------------|
| **Penalty**          | Sum of the absolute values of the weights              | Sum of the squared values of the weights               |
| **Effect on Weights**| Encourages sparsity; some weights may become zero      | Shrinks weights but typically does not make them zero  |
| **Use Case**         | Feature selection; promoting simpler models            | General overfitting reduction; maintaining all features|


L2 Regularization
: ![img_8.png](img/img_8.png)

L1 Regularization
: ![img_9.png](img/img_9.png)


Weight decay
: penalizing large weights , promoting simpler & more generalizable models 
: effectively constrains the model's capacity to fit noise
: Weight decay modifies the weight update rule directly,
whereas L2 regularization adds the penalty term to the loss function.
: weight decay = L2 regularization are mathematically similar BUT   differently in optimization

![img_10.png](img/img_10.png)


Early stopping 
: stop model training when : - training error decreases and validation error rise again.
: stop when for n epochs validation accuracy not improved 
: so improve overfitting and reduces training 
![img_14.png](img/img_14.png)


Regularization : Dropout
: randomly drop units(dropout rate 20-50%) during training 
: reduces overfitting and improves generalization 

![img_15.png](img/img_15.png)

why Dropout is working?
: prevent co-adaption develop independent feature detectors
: Combining these subnetworks during inference resembles an ensemble approach

parameter tying 
: share parameters (weights) across multiple parts of a model
: reduce # of parameters  , leading to model that generalize better => less overfitting risk

![img_1.png](img/i/img_1.png)
![img_2.png](img/i/img_2.png)

feature scaling
: preprocessing technique  to normalize or standardize the range of features in a dataset
: Improves convergence - due to  similar scale
: prevents dominance
: reduce computational complexity 

Feature scaling techniques:
: normalization 
: standardization 
: robust scaling 
: unit vector scaling 

norm vs unnorm 
:  ![img_16.png](img/img_16.png)
: help GD for glo bal minima more quickly 


Batch normalization 
: implement as layer
: norm layer inputs to stabilize and accelerate training &  generalizes better
: after the linear transformation(perceptron formula) and before the activation function within a layer.
: Perc->BN->Actv

Batch normalization : normalization 
: compute  the mean  and variance of the inputs.
: Normalize the inputs to have zero mean and unit variance:
: Scaling and Shifting : allow the network to adjust the normalized output:
: γ (scale) and β (shift)

![img_4.png](img/i/img_4.png)

![img_5.png](img/i/img_5.png)

![img_6.png](img/i/img_6.png)


mini batch size
: size= m : takes too much time  for iteration 
: size=1 :  extremely noisy  , more time to reach the global minima 

![img_19.png](img/img_19.png)


# optimization 

Gradient descent 
: use entire dataset
: minimizing the loss function to improve accuracy 
: and optimizing model parameters to generalizes wll to new unseen data  
: challenging in non-convex to not get trapped in local minima or saddle points
: challenging to choose the lr with better convergence

: gradient/partial derivative of the loss function with respect  to each parameter 
: then adjust each parameter 

![img_20.png](img/img_20.png)

GD Categories:
: Batch GD.
: Stochastic GD.
: Mini-Batch GD.

GD Techniques to  Improve Gradient Descent:
: Adam,
: Momentum,
: Learning Rate Scheduling.
: RMSProp
: NAG
: Adagrad

Mini-batch GD
: small batched of data 
: cause much faster training 
: mini patches allows parallel computation and reduce  memory requirements 
: Balancing Noisy and Stable Updates than SGD
: shuffle data in each batch 

SGD: Stochastic GD
:  simple input example  randomly  to prevent overfitting
: better generalization
: help to escape local minima or saddle point  
:  cause fluctuations in loss function 
: very fast , lower memory req 


| **Aspect**            | **Stochastic Gradient Descent (SGD)**                                    | **Mini-Batch Gradient Descent**                                  |
|-----------------------|--------------------------------------------------------------------------|------------------------------------------------------------------|
| **When to Use**       |                                                                          |                                                                  |
| **Dataset Size**      | Small datasets or streaming data (online learning).                     | Large datasets that benefit from batch processing.               |
| **Convergence Goal**  | Escaping local minima or saddle points in non-convex loss landscapes.    | Faster and more stable convergence with balanced noise.          |
| **Hardware**          | Limited resources (e.g., CPU-based training or low memory).             | Leveraging GPUs for parallel processing with mini-batches.       |
| **Task Type**         | Online or real-time learning where data arrives sequentially.            | Deep learning models or tasks requiring gradient stability.       |
| **Implementation**    | Simple or iterative models without batching requirements.               | Scenarios needing balance between computational efficiency and accuracy. |
| **When Not to Use**   |                                                                          |                                                                  |
| **Dataset Size**      | Large datasets; frequent updates become computationally expensive.       | Extremely small datasets where batching adds unnecessary overhead. |
| **Convergence Speed** | When stable and fast convergence is crucial (e.g., production-ready models). | When low noise is essential for precise optimization (e.g., highly structured data). |
| **Hardware**          | High-performance GPUs; does not fully utilize their parallel capabilities. | Limited resources where batch memory requirements cannot be met. |
| **Task Type**         | Large-scale training with complex models that require stable gradients.  | Streaming or real-time data, where batches are impractical.      |


Momentum and GD
: help with escaping Local Minima
: accelerates convergence by smoothing out oscillations in the optimization process.
: is an enhancement to Gradient Descent
: Adds a fraction of the previous update to the current one, carrying forward the direction of the previous step.
: Helps the optimization algorithm build speed in directions with consistent gradients.
: Smoother Updates

![img_23.png](img/img_23.png)

Nesterov Accelerated Gradient (NAG)
: leading to more accurate updates 
: helps to avoid overshooting 

![img_9.png](img/i/img_9.png)

NAG vs momentum
: Momentum: Effective in general cases, especially for smooth, convex problems.
: NAG: Preferred when overshooting is a concern or for non-convex loss surfaces, 
        as it provides better control over the optimization path.


![img_7.png](img/i/img_7.png)

![img_8.png](img/i/img_8.png)

Vanishing/exploding gradient 
: The vanishing gradient problem occurs in neural networks when gradients
    become extremely small as they are backpropagated through layers.
: This slows down or stalls learning for earlier layers.

ADAM adaptive moment estimation 
: normalized gradient , use 1/2 order 
: prevent vanishing gradient problem 
: use methods: 
: momentum : moving average  of past gradient to accelerate convergence 
: AdaGrad : adapts lr for each parameter based on past gradients


# Deep Learning Training Order
A Deep Learning Neural Network (DL NN) is a computational model that uses feedforward propagation to pass inputs through layers of neurons, generating predictions. During training, the model employs backpropagation to calculate the error gradient with respect to each weight, which is then minimized using gradient descent, an iterative optimization technique. To enhance the model's performance and prevent overfitting, techniques such as regularization are applied, which constrain the model complexity during the optimization process.

## Steps with Feedforward and Backpropagation

1. **Training Phase**:
    - **Feedforward**:
        - Input data from the **training set** is passed through the neural network to compute predictions.
    - **Loss Calculation**:
        - Compute the loss between predicted outputs and actual labels.
    - **Backpropagation**:
        - Calculate gradients of the loss with respect to weights and biases.
        - Update weights and biases using an optimization algorithm (e.g., SGD, Adam).

2. **Validation Phase**:
    - Use the **validation set** to evaluate model performance.
    - Metrics like accuracy or loss are monitored without updating the weights.
    - Used for:
        - Early stopping.
        - Hyperparameter tuning.

3. **Testing Phase**:
    - After training is complete, evaluate the model on the **test set**.
    - This assesses performance on unseen data without updating weights.

## Summary:
- **Training set**: Used with feedforward and backpropagation.
- **Validation set**: Used for evaluation during training (no weight updates).
- **Test set**: Used after training for final performance evaluation.


why we don't use validation set to update the weights? 
: purpose is to evaluate the model's performance on unseen data during training,
: not to directly influence the training process. 








