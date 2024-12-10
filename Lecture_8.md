# Recurrent NN memory and attention 

Residual Neural Networks (ResNets) and Recurrent Neural Networks (RNNs):
: Residual: For static data; focuses on learning residuals to enable very deep architectures.
: Recurrent: For sequence data; focuses on modeling temporal dependencies.

RNN recurrent neural networks:
: learn streams , sequential data 
: temporal dependent e.g. time-series , raw sensor  , text , image , video 
: can process variable-length sequences 


![img_27.png](img/img_27.png)

![img_36.png](img/img_36.png)

examples
: activity  recognition from  sensors
: machine translation , Q&A , opinion mining 
: speech recognition & generation
: image captioning   , video activity recognition 

RNN   I/O . 
: 1-1 (vanilia) : classification
: 1-m: image captioning  , sequence of words
: m-1: text sequence binary classification , sentiment classification 
: m-m: video classification, machine translation 

: ![img_28.png](img/img_28.png)

neuron architecture  - vanilla rnn cell 
: tanh for non-linearity 
: Encodes past information to maintain temporal relationships in sequential data.
: concatenation previous hidden state and input

 ![img_29.png](img/img_29.png)
 ![img_34.png](img/img_34.png)


Steps
: ![img_12.png](img/i/img_12.png)
: ![img_13.png](img/i/img_13.png)
: ![img_14.png](img/i/img_14.png)
: *** Hidden-to-Hidden Weights (𝑈ℎ)
: *** Input-to-Hidden Weights (Wh)


Unrolloing  RNN
: RNN can be thought as multiple copies of the same Network each passing a message to a successor
: A vanilla RNN uses the same structure (same weights and biases) across all time steps.

: ![img_35.png](img/img_35.png)
: ![img_33.png](img/img_33.png)
: ![img_11.png](img/i/img_11.png)

self-supervised
: The model is trained to predict the next term in the sequence using the current term as input
: no explicit target sequence :  the model generates its own targets based on the input data 
: e.g. Input: "The cat is on the" : Target: "cat is on the mat"
: an advance of 1 step 

memory-less models for sequences  (not RNN)
: feed-forward neural nets : each output is computed directly from the current input  without explicitly maintaining a hidden state or temporal relationship.
: they do not retain or use info from prev time steps explicitly , 
: Autoregressive models : predict next term from fixed # of previous terms (not retain memory)

RNN vs Memory-less models: 
: Memory-less:  Only consider immediate input or a fixed window of past inputs, 
    without retaining internal states.
: RNN: Retain hidden states that capture temporal dependencies,

hidden states to retain information over time
:  allows it to store information for longer durations,
: == making it capable of capturing temporal dependencies/patters .
: if output is noisy , hidden state cannot be determined precisely ->
    The best approach is  probabilistic methods.


Linear Dynamical Systems (not RNN)
: use for time-series data with hidden states.
: hidden states : real-valued , with linear relationships between variables
: gaussian noise influence output and hidden states 
: used by apps where the system has predictable dynamics influenced by noise, such as tracking or control systems.


Hidden Markov Models (HMMs) (not RNN)
: probabilistic model used to handle sequences with hidden states.
: The outputs depend probabilistically on the hidden state but are not deterministic.
: Practical Use:  Predict the probability distribution over hidden states to make sequential predictions.
: Limited Memory, log(N), N= Hidden states
: Long-term Dependencies: complex temporal relationships can not be handled 

RNN features & capabilities: 
: distributed hidden states: store/process info from past
: Non-linear Dynamics: capture patters in sequential data
: flexibility for solving complex sequence-based problems.

Stochastic vs. Deterministic Models& hidden states:
: Stochastic (HMMs, LDS) use randomness to infer or predict outcomes. They focus on modeling the probability distributions of sequences or events.
: Deterministic  (RNNs) use a fixed transformation (non-random) to process input data and hidden states.



RNN Behavior(some problematics)
: ** the problematics not happen every time 
: Oscillation:  , cycling through specific patterns of hidden states. 
: Point Attractors:  hidden state may converge to a stable fixed point (point attractor), 
    even with minor input variations. Once in this state, the RNN stops changing and effectively "settles."
: Chaotic Behavior: small changes in inputs or hidden states can result in
        exponentially diverging outputs, leading to unpredictable and chaotic behavior.
: Parallel Processing:  RNNs can act as small parallel programs, 
     different units  capturing specific patterns.
: Dynamic Adaptation: RNNs dynamically update their internal state
    to adapt to changing patterns in sequential data over time,
        making them effective for modeling time-varying dependencies.
: Long-term Dependencies:designed to maintain information over long sequences, 
    capturing dependencies across extended time spans



RNN  Challenges
: vanishing/exploding gradients 
: high computational demands.
: sensitivity to initialization 
: careful training 


RNNs Share Weights
: Temporal Consistency, Parameter Efficiency, Training Simplicity, Improved Generalization:


RNN specify 
: Inputs: Can be specified as initial states, states for subsets, or states at every time step.
: All the hidden states are initialized with specific values->  fresh without prior context,
: Targets: Can focus on final states, specific time steps, or a subset of units depending on the task.
: This flexibility allows RNNs to handle a wide variety of sequential tasks, such as language modeling, time-series forecasting, and sequence-to-sequence translation.



Characteristics of Chain RNNs
: processes a sequence of inputs in a step-by-step manner over time. 
: Sequential processing 
: temporal dependencies
: shared weights 

RNN issues: 
: struggle with long-term dependencies => capture phrases  between elements that are far apart in the sequence.
: focus on final vector => struggle in equal attention  
: one-side context 

![img_42.png](img/img_42.png)

Truncated backpropagation  & forward: 
: through chunks instead of whole sequence 

RNN vanishing
: Gradients are computed at each time step and multiplied recursively to update weights.
: The chain rule involves multiplying derivatives of W, which can cause the gradient to shrink (or grow) exponentially.
: W contains small values (e.g. ∣W∣<1) ->  vanishing gradients. = early zero updates, making it hard for the model to learn long-term dependencies.
: W contains large values (e.g. ∣W∣>1) ->  exploding gradients Gradients grow exponentially,

---


LSTM
: Long short-term memory 
: for short-term memory : hidden state 
: for long-term memory: cell state 
: keep-write-read = forget-input-output

![img_43.png](img/img_43.png)

![img_44.png](img/img_44.png)

![img_15.png](img/i/img_15.png)
![img_16.png](img/i/img_16.png)
![img_17.png](img/i/img_17.png)
![img_18.png](img/i/img_18.png)
![img_19.png](img/i/img_19.png)

LSTM prevent van/exp gradient
: additive updates :  Instead of multiplying the state repeatedly , adds updates using the forget and input gates.
: activation functions: sigmoid and tanh  :bounding , scaling updates and stabilizing gradients  
: Gates: 

| **Feature**       | **Addresses**                | **How It Helps**                                                                 |
|--------------------|------------------------------|----------------------------------------------------------------------------------|
| **Cell State**     | Vanishing Gradients          | Allows gradients to flow with minimal decay across long sequences.               |
| **Forget Gate**    | Vanishing Gradients          | Prevents irrelevant information from accumulating in the cell state.             |
| **Input Gate**     | Vanishing Gradients          | Controls how much new information is added, reducing disruptive updates.         |
| **Output Gate**    | Exploding Gradients          | Regulates the output, ensuring gradient values remain within a manageable range. |
| **Gradient Clipping** | Exploding Gradients          | Prevents excessively large gradients during backpropagation.                  |



LSTM prevent van/exp gradient why it works? 
: controlled info flow : gates selectively allow relevant info while discarding irrelevant or noisy data 
: stable gradient propagation : additive cell stable gradient over long seq 

![img_48.png](img/img_48.png)
![img_47.png](img/img_47.png)

![img_49.png](img/img_49.png)
![img_45.png](img/img_45.png)


LSTM Training Workflow
: Forward Pass:
    1. Pass input sequences through LSTM.
    2. Compute gates, cell states, and hidden states.
    3. Generate predictions.
: Loss Calculation: 
        1. Compare predictions to ground truth.
        2. Backward Pass (BPTT):
        3. Compute gradients for weights using the loss.
: Weight Update: Use optimizers to update weights.
: Repeat: Iterate through epochs until convergence.

LSTM addresses: 
:  1. Long-Term Dependencies : cell state  allow info to persist across long time 
: 2. Irrelevant Information Accumulation : forget gate selectively discards irrelevant info 
: 3. Overwriting Important Information: input gate regulate how much new info is added 
: 4. Difficulty Handling Nonlinear Dependencies : gating mechanisms introduce nonlinear transformations 
: 5. Learning Dynamic Temporal Patterns: process sequences element by element,  suitable for varying-length and dynamic patterns
: 6. Gradient Flow Regulation : output gate regulates the gradient flow to the next layer by squashing the output with activation functions like tanh ⁡
: 7. Handling Sequential Data with Missing Values :  learn to ignore missing values or interpolate through time steps,


sequence labeling 
: Assigning a label to each element in an input sequence.
: Input Data: Sequences of input data, e.g., word embeddings for text or feature vectors for time-series data.
: loss function : Use a sequence-level loss function like Cross-Entropy
: Compute gradients across the sequence using Backpropagation Through Time (BPTT)


BP vs BPTT
: BPTT : Considers both the spatial (layer-by-layer) and temporal (time-step-by-time-step) dependencies.
: BP : No temporal or sequential dependencies.

| **Aspect**                 | **Backpropagation (BP)**               | **Backpropagation Through Time (BPTT)**   |
|----------------------------|----------------------------------------|-------------------------------------------|
| **Network Type**           | Feedforward networks                  | Recurrent networks (RNNs, LSTMs, GRUs)    |
| **Handles Sequences?**     | No                                    | Yes                                       |
| **Steps**                  | Gradients propagated through layers.  | Gradients propagated through layers **and time steps**. |
| **Vanishing/Exploding Gradients** | Less significant in shallow networks. | Major issue, mitigated by LSTMs/GRUs.    |
| **Unrolling**              | Not required.                         | Requires unrolling the RNN over time.    |
| **Computation**            | Less memory-intensive.                | More memory-intensive (due to unrolling). |


Bi-directional LSTM 
: processes input sequences in both forward and backward directions  (Two Separate LSTMs ) 
: concatenate both hidden state 
:  Bidirectional Context : past, future 

GRU: gated recurrent units 
: Faster Training: Simpler architecture with fewer parameters.
: address vanishing gradient and log-term dependency( to some extent )
: Efficient for Short Dependencies
: combine gate ( forget & input) into update gate ->reduce the num of parameters 

Seq2Seq - sequence-to-sequence 
:  with RNNs or LSTMs
: Encoder-Decoder 
: Pre-Attention

Encoder
: encode input into fixed-length vector representation 

Decoder
: takes from encoder and generate the output seq  



