# CNN pushing the limits of depth 


Training  vs inference 
: training : backpropagation using SGD  
: Inference : Determine the class of new input 

Training Instability
: Very deep architectures often suffer from training instability due to vanishing or exploding gradients,
: but techniques like dropout and batch normalization stabilize training by regularizing activations and normalizing intermediate outputs.


Data augmentation
: horizontal flip 
: random cropping and rescaling 
: color jittering e.g. brightness , contrast , saturation 
: geometric distortions e.g. rotations


data augmention 
: when: only to the training set 
: why : fair evaluation to generalize to unseen data 


Normalization  
: balance varying ranges to help converge faster in training 
: during preprocessing before NN

batch normalization  
: between layers for stabilize intermediate activations  
: Conv -> BatchNorm -> Activation 
: Dense -> BatchNorm -> Activation 
: ensure mean=0 , variance=0 
: reduce exploding/vanishing gradients 
: speedup convergence allowing higher lr 
: reducing overfitting adding slight regulation effect 

Transfer learning 
: 2 types : feature extraction  & gine-tuning
: when a model trained on a different task is reused for a new related task
: reduce training time 
: improve performance 
: helpful in limited datasets 

Transfer learning decision
: Start with Feature Extraction for smaller datasets and low computational needs.
: Use Fine-Tuning for larger datasets or when the task/dataset differs significantly from the original model's training.

Feature extraction 
: pre-trained model as a fixed 

Fine - tune:
: adjust wights  of the rpe-trained model
: Similar datasets: Fine-tune only the top layers.
: Dissimilar datasets: Fine-tune the entire network progressively.

![img_10.png](img/im/img_10.png)

![img_11.png](img/im/img_11.png)

Multitask learning (MTL)
: model trained to perform simultaneously related task 
: why:  improve generalization , limited data 
: when: related tasks , same input data 
: leverage shared :  knowledge(features)  &  architecture( some layers, usually top) 
: parallel training , share loss function 

![img_9.png](img/im/img_9.png)

Networks from slides:
: widely  used for transfer learning

| **Network**   | **Category**                | **Why Use It**                                                                                      |
|---------------|-----------------------------|----------------------------------------------------------------------------------------------------|
| **AlexNet**   | Classic CNN                 | Suitable for small-to-medium datasets with simple fine-tuning, as it introduces foundational CNN techniques. |
| **VGGNet**    | Very Deep CNN               | Ideal for tasks requiring deeper architectures; feature extraction is effective due to uniform filters. |
| **GoogleNet** | Inception-Based Architecture | Efficient and powerful for tasks with computational constraints, using Inception modules effectively. |
| **ResNet**    | Residual Network            | Best for very deep tasks; fine-tuning is reliable due to skip connections mitigating gradient issues. |

![img_5.png](img/img_5.png)

![img_3.png](img/img_3.png)

![img_4.png](img/img_4.png)

ResNet 
: skip connections to address vanishing/exploding gradient problem  
: enabling training of extremely deep network

GoogleNet
: reduce computation with bottleneck layers 
: multi-scale feature extraction by combining multi-size filters

VGGNet:
: smaller uniform filters without adding excessive parameters 

AlexNet :
: need for Data preprocessing e.g. normalization .
: introduce ReLu, dropout 
: data augmentation to combat overfitting 


Challenges of Deep Networks
:   Vanishing/exploding gradients in traditional architectures.
:   Overfitting and the need for regularization techniques (e.g., dropout, batch normalization).



--- 

Residual networks
: address vanishing/exploding gradient that be caused by deeper architectures.
: introduce residual blocks to skip connections in order to preserve information 
    if the specific block fail to learn useful transformation 
        and the initial input ( of the block) is still carried forward.

Structure of Inner Layers in Residual Blocks
: Conv→BN→ReLU→Conv→BN
: 1x1 Conv (Reduce)→BN→ReLU→3x3 Conv (Extract)→BN→ReLU→1x1 Conv (Expand)→BN

Residual mappings formula:
: F(x)=H(x)−x

---



Inception-Based Networks
: Parallel Feature Capture: Simultaneously apply multiple operations.
: Multi-Scale Convolutions & Pooling: Learn spatial features at various granularities with different kernel sizes (1x1, 3x3, 5x5).
: 1x1 Convolutions for Dimensionality Reduction: Reduce computational and memory requirements without losing essential information.
: Filter Concatenation: Merge outputs from multiple operations into a single dense representation.
: Layer Stacking: Increase network depth while maintaining manageable computational complexity.

![img_13.png](img/im/img_13.png)



![img_12.png](img/im/img_12.png)

Problems Inception Solves
: Computational Efficiency: Reduced number of parameters .
: Overfitting: Reduced parameters and dimensionality 
: Feature Diversity: Multi-scale convolutions capture diverse feature representations.
: Scalability: easy extension and adaptation.
: Improved Accuracy: Achieves high performance on classification tasks (e.g., ImageNet).





















