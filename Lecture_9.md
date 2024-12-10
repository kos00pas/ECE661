# Transformers and attention



Encoder
: encode input into fixed-length vector representation

Decoder
: takes from encoder and generate the output seq

Attention
: seq-to-seq
: encoder-decoder architecture
: encoder compress input to single fixed-length
: decoder selectively focus of different parts of input - dynamically focus


![img_50.png](img/img_50.png)

![img_51.png](img/img_51.png)

Attention in Encoder-Decoder:
:  With attention, the decoder queries the encoder at every time step to
        decide which parts of the input sequence are most relevant to the current decoding step

Attention Mechanism:
: overcome single fixed-length context vector
: dynamically focus on relevant parts of the input sequence during decoding
: Instead of relying only on the final hidden state of the encoder, attention computes a weighted combination of all encoder states, allowing the decoder to access detailed information at every step.


Vanilla attention
: improves the encoder-decoder architecture by allowing the model to dynamically focus on 
        relevant parts of the input, especially in tasks involving long or complex sequences.

difference between attention and the encoder-decoder
: lies in the dynamic focus on the input(attention)
: Encoder-Decoder: Compresses  input  into a fixed-length context vector.  The decoder relies on  fixed context vector to generate the output sequence
: Attention: Allows the decoder to dynamically focus on different parts of the input sequence at each decoding step
Instead  single context vector, attention computes a weighted sum of all encoder hidden states based on relevance.


image caption generation using attention
:  attention  identifies region contributes the most to the next caption word. 
: assign weights to regions
: weighted sum calculated and use to generate the first word of the caption
: Calculating updated attention weights and start again for the next one

Attention :
: focus on different image regions iteratively
: Dynamically update parameters to generate meaningful captions

Is attention same as transformer????
: No, transformer heavily relies on attention
:  Attention:  mechanism that enables the model to **focus on specific parts of the input** sequence when making predictions.
:  Transformer: It completely replaces RNNs and CNNs **with attention** mechanisms for processing sequences.
Relies entirely on **self-attention** (scaled dot-product attention) to capture dependencies within a sequence.
Uses **multi-head** attention for parallel processing and richer representation.

Transformers
: architecture introduced in the paper "Attention Is All You Need" (2017).
: Self-Attention Mechanism: focus on different parts of input for each output element
: Layer  normalization : stabilize and improve training
: MLP (multi-layer-perceptron):
: Scalability: Ideal for large-scale models due to its parallelism
: Highly memory-intensive

---

Receptive field
: region of the input image that a specific output in influenced by

type of attention
: **self-attention** : input attends to itself
: **encoder-decoder** : seq-to-seq , the decoder attends to the encoder's output to align input and output sequences.
: **multi-head** : multiple attention in parallel

Attention importance
: dynamic focus : prioritize relevance  to improve efficiency
: handles long-range dependencies : distant data affect each other
: scalability: parallel processing


embeddings
:  are the vector representations of parts of the input data.


permutation invariance
: the order of inputs does not affect the output.
: the model treats the input as a set (unordered collection) rather than as a sequence or grid with a specific order

![img_1.png](img/_img_1.png)


btw
: MSA = (Multi-Head Self-Attention)

---

# Transformer


![img_22.png](img/i/img_22.png)


Workflow
: Encoder
: Decoder
: Final output

A. Encoder
: 1 Input Embedding : 2 Positional Encoding : 3 Multi-Head Attention: 4 Add & Norm: 5  Feed-Forward: 6 Add & Norm : 7 Repeat N-times.

B. Decoder
: 8 Output Embedding: 9  Positional Encoding: 10 Masked Multi-Head Attention: 11 Add & Norm: 12 Encoder-Decoder Attention: 13 Add & Norm: 14 Feed-Forward: 15 Add & Norm: 16 Repeat N-times.


C. Final Output
: 17 Decoder Output: 18 Linear: 19 Softmax: 20 Output Probabilities.



---

Goal of Encoder
: 

> extract contextual representations  that captures the relationships and dependencies between all elements of the input
 
>(This representation is then used by the decoder to produce the output sequence.)


Goal of Decoder 
: 

> generate the output sequence step-by-step, 

>  leveraging both the contextual representations from the encoder and its own previously generated outputs

Goal of Add & Norm
: 

> Add ->  Residual Connections -> Helps gradient flow -> reducing the risk of vanishing gradients

> Norm:  Stabilizes training by reducing the sensitivity to changes. 
         Ensures consistent input magnitudes for downstream layers, improving convergence speed.


Goals of Self-Attention
: 

>  the model to capture relationships between all tokens in the sequence, regardless of their positions.

Goal of Multi-head 
: 

> enable the Transformer to focus on different parts or aspects of the input sequence
>           simultaneously and independently,

> improving its ability to capture diverse relationships and dependencies within the data.

Q (Query), K (Key), and V (Value) matrices

: Αre calculated by applying learnable linear transformations to the input embeddings
: ![img_34.png](img/i/img_34.png)
: Values (V): The actual content (information) of each region.
: Keys (K): What information does each region contain?
: Queries (Q): What information are we looking for?

---

## A. Encoder

1 Input Embedding
: input raw tokens convert into dense vectors using an embedding layer
: ![img_23.png](img/i/img_23.png)

2 Positional Encoding
: Adds positional information to the embeddings to capture sequence order
: needed for parallel processing
: ![img_27.png](img/i/img_27.png)

3 Multi-Head Attention
: Computes attention across all tokens
: ![img_24.png](img/i/img_24.png)

4 Add & Norm
: Adds the original input (residual connection) to the attention output and applies layer normalization.
: ![img_25.png](img/i/img_25.png)


5  Feed-Forward
: Applies a fully connected network to each token independently
: ![img_26.png](img/i/img_26.png)

6 Add & Norm
:  Adds the output of the feed-forward network to the input and applies layer normalization.
: ![img_28.png](img/i/img_28.png)

7 Repeat N-times.
: steps (3–6) are repeated N-times in stacked layers.

## B. Decoder

8 Output Embedding
: Target sequence tokens  are converted into dense vectors using an embedding layer
: ![img_29.png](img/i/img_29.png)


9  Positional Encoding
: Adds positional information to the output embeddings

10 Masked Multi-Head Attention
:  Self-attention is applied with a mask to prevent attending to future tokens.
: ![img_30.png](img/i/img_30.png)

11 Add & Norm
: Adds the output of masked attention to the input and applies layer normalization.
: ![img_25.png](img/i/img_25.png)

12 Encoder-Decoder Attention
: The decoder attends to the encoder’s outputs using standard attention.
: ![img_31.png](img/i/img_31.png)


13 Add & Norm
: Adds the encoder-decoder attention output to the decoder’s input and applies layer normalization.


14 Feed-Forward
: as in the encoder.

15 Add & Norm
: Adds the output of the feed-forward network to the input and applies layer normalization.


16 Repeat N-times.
:  The above steps (10–15) are repeated N-times in stacked layers.

## C. Final Output

17 Decoder Output
:  The final output from the decoder is a sequence of vectors for each token

18 Linear
: Projects the decoder output into the vocabulary size.
: ![img_32.png](img/i/img_32.png)


19 Softmax
: Converts the logits into probabilities for each token in the vocabulary
: ![img_33.png](img/i/img_33.png)

20 Output Probabilities.
: The final output probabilities are used to predict the next token or complete the sequence



---


How the Number of Meaningful Regions Is Determined
:  it assigns higher weights to the regions most relevant to the current task while reducing the influence of less relevant regions.
: Dynamically adjusting weights using softmax normalization.

Similarity Calculation:
: For each region, the similarity between its Key and the Query is computed using a dot product
: This measures how relevant each region is for the current decoding step.

---


Transformers address

1. Vanishing and Exploding Gradients (in RNNs)

> Transformers process sequences in parallel instead of step-by-step, removing the dependency on sequential gradient propagation.

2. Long-Range Dependencies ( RNNs and LSTMs )
> The self-attention mechanism allows each token to attend to all other tokens in the sequence,
        enabling effective modeling of both short- and long-term dependencies.


 3. Sequential Processing (RNN Bottleneck)

>  Transformers use self-attention, enabling parallel computation for the entire sequence, significantly speeding up training and inference.

 4. Fixed-Length Representations (in Encoder-Decoder Models)

>  The attention mechanism dynamically computes a context vector for each output step, allowing access to all encoder states.


 5. Scalability and Performance ( RNNs and LSTMs are computationally expensive)

> The attention mechanism dynamically computes a context vector for each output step, allowing access to all encoder states.

 6. Limited Representational Power (CNN ,RNN)

>  Multi-head attention allows the model to focus on multiple aspects of the input simultaneously, capturing diverse relationships and dependencies.

 7. Dependency on Positional Information (RNN)
 
> Transformers introduce positional encoding to provide sequence order information to the model.

Summary : Transformers address the following challenges:
: Vanishing/exploding gradients.
: Modeling long-range dependencies.
: Slow sequential processing in RNNs.
: Information loss in fixed-length representations.
: Scalability for large datasets and long sequences.
: Generalization across tasks.
: Dependency on sequence order.


| **Problem Addressed by Transformers**     | **Other Architectures with These Problems** |
|-------------------------------------------|---------------------------------------------|
| Vanishing/Exploding Gradients             | RNNs, LSTMs                                 |
| Long-Range Dependencies                   | RNNs, LSTMs, GRUs                           |
| Sequential Processing Bottleneck          | RNNs, LSTMs                                 |
| Fixed-Length Representations              | Seq2Seq with RNNs, LSTMs                    |
| Scalability to Large Datasets/Sequences   | RNNs, LSTMs, CNNs                           |
| Limited Representational Power            | RNNs, LSTMs, CNNs                           |
| Lack of Positional Information            | CNNs, Feedforward Networks                 |
| Task-Specific Architectures               | Traditional NLP Models                      |
| Inefficient Data Parallelism              | RNNs, LSTMs                                 |


