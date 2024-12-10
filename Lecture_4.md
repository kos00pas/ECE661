

rule of thumb
: # of training 
: at least 5-10  
: the # of weights of the network 

universal theorem 
: a feedforward network approximates a function 
: any continuous function can be realized by a network with one hidden layer 
: (given enough hidden neurons )


why then  have multi hidden  layers? 
: may fail to learn & generalize correctly 
: using deeper models :
1.  reduce # of units required to represent the function 
2. reduce the amount of generalization error 
: Deeper networks require fewer neurons per layer to achieve the same level of approximation
: deeper nw: fewer units per layer and far fewer parameters  
![img_1.png](img/img_1.png)

![img_2.png](img/img_2.png)


DL challenges 
: vanishing/exploding  gradients 
: choose activation function 
: init weights 
: architecture 
: hyper-parameters 


| **Activation Function** | **Best Use Case**                      | **Output Range**        |
|--------------------------|----------------------------------------|-------------------------|
| **ReLU (Rectified Linear Unit)** | - Most commonly used in hidden layers. <br>- Works well in deep networks by mitigating the vanishing gradient problem. | \([0, \infty)\)         |
| **Leaky ReLU**          | - Preferred over ReLU to prevent "dying neurons" (neurons stuck with zero gradients). | \((-\infty, \infty)\)   |
| **Sigmoid**             | - Suitable for binary classification in the output layer. <br>- Avoid in hidden layers due to vanishing gradients. | \([0, 1]\)              |
| **Tanh**                | - Often used in shallow networks where centered outputs (\([-1, 1]\)) improve convergence. | \([-1, 1]\)             |
| **Softmax**             | - Used exclusively in the output layer for multi-class classification tasks. <br>- Ensures outputs represent probabilities (sums to 1). | \([0, 1]\) (sums to 1)  |

![img_6.png](img/im/img_6.png)

![img_3.png](img/im/img_3.png)

![img_4.png](img/im/img_4.png)

![img_5.png](img/im/img_5.png)

![img_7.png](img/im/img_7.png)

![img_8.png](img/im/img_8.png)

















