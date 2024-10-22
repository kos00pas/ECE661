# Assignment 3 
## Pashiourtides Costas 

# Exercise 1

## Part 1
### What is RNN 
- RNN stands for Recurrent Neural Network and is a type of deep neural network.
- It is designed to handle sequential or time-series data.
1. 
   - The current output depends on previous computations by **passing the hidden state** from one time step to the next.
   - The hidden state retains information about **previous inputs**, allowing the network to process the current input with the context of past data.
   - This means that RNNs have memory and can 'remember' previous data points, making this the key to dealing with sequence data.
2. 
   - **share the same weight parameters** across all time steps 
   - This allows the  model to treat each part of the sequence equally,
     - making it capable of learning patterns that occur at any point in the sequence.
3. .
   - RNNs are trained using **Backpropagation Through Time (BPTT)**.
     - BPTT unrolls the RNN across all time steps and computes gradients for each time step. 
     - These gradients are then used to update the weights

### RNN Differ from Feedforward NN

1. **Output Dependency**  
   - **Feedforward NN**: The output is only dependent on the current input.  
   - **RNN**: The output depends on both the current input and previous computations via the hidden state.

2. **Memory of Past Inputs**  
   - **Feedforward NN**: Has no memory; each input is processed independently.  
   - **RNN**: Retains memory through hidden states, allowing the model to "remember" past inputs.

3. **Weight Sharing**  
   - **Feedforward NN**: Each layer has unique weight parameters, treating each input as independent.  
   - **RNN**: Shares the same weight parameters across all time steps, enabling it to generalize across the entire sequence.

4. **Training Process**  
   - **Feedforward NN**: Trained using standard backpropagation.  
   - **RNN**: Trained using **Backpropagation Through Time (BPTT)** to compute gradients over all time steps.

## Part 2
### Vanishing gradient problem 

### Vanishing gradient problenm in RNN 

### WHY vgp affect training  

## Part 3
### RNN affectivenes through application  
#### Application 1:

#### Application 2: 

## Part 4
