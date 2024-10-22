# Assignment 3 
### Pashiourtides Costas 

# Exercise 1

## Part 1 ----------------------------------------------------------------------
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
3. 
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

## Part 2 ----------------------------------------------------------------------
### Vanishing gradient problem 
- In backpropagation, when the gradient approaches zero, it makes weight updates insignificant, preventing effective learning in earlier layers.

### Vanishing gradient problem in RNN 
- Due to sequential nature of RNN gradients are propagated not only through layers but also through time steps making long-term dependencies hard to learn.


### Why Vanishing gradient problem affect RNN  training  
- Vanishing gradients result in minimal or no weight updates, preventing the network from effectively learning and capturing important patterns.
- Due to RNN's recursive nature, gradients are propagated through both layers and time steps, which causes long-term dependencies to vanish more quickly.


## Part 3 ----------------------------------------------------------------------
### Application 1: Speech Recognition
- RNNs are effective in speech recognition as they can process sequences of audio signals over time, capturing the temporal dependencies between phonemes and words to convert spoken language into text.
### Application 2: Video Analysis
- RNNs are used in video analysis to model temporal relationships between video frames, enabling tasks such as action recognition, video captioning, and anomaly detection by understanding the sequence of events over time.

## Part 4 ----------------------------------------------------------------------



# Exercise 2
## Part 1 ----------------------------------------------------------------------
### LSTM
- LSTM is a type of RNN designed to handle long-term dependencies in sequence data.
- It has a more complex architecture with forget, input, and output gates, allowing it to selectively retain or forget information over long sequences.
### Addressing Vanishing Gradient Problem
- The gate mechanism helps prevent the gradient from becoming too small during backpropagation.
  - How? By regulating the flow of information in the network to preserve essential data and maintain strong gradients:
    - Forget Gate: Decides what information to discard from the cell state.
    - Input Gate: Determines what new information to add or update in the cell state.
    - Output Gate: Controls what information is output as the hidden state at each time step.
    - Cell State: allow information to flow through without being significantly modified, preserving important information over time.
## Part 2 ----------------------------------------------------------------------
### LSTM examples with superior performance to RNNs
- Speech Recognition
  - More accurate and reliable in capturing context over long speech sequences, improving recognition accuracy.
- Text Generation
  - Can generate more contextually relevant and structured sentences, keeping track of long-term dependencies.
- Machine Translation
  - Performs better with longer and more complex sentences, improving translation accuracy.


## Part 3 ----------------------------------------------------------------------




