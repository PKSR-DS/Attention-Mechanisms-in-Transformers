# Attention-Mechanisms-in-Transformers

Overview
This repository contains a Python implementation of attention mechanisms in Transformer models. It covers the concepts of self-attention and multi-head attention, two key components in transformer architectures.

Key Concepts
Self-Attention: The self-attention mechanism allows the model to focus on different parts of the input sequence to understand the context better. Each word in a sentence attends to every other word, helping the model capture the relationship between them.

Multi-Head Attention: This extends the self-attention mechanism by splitting it into multiple "heads" that process different parts of the input sequence simultaneously. This allows the model to capture a richer set of relationships.


Requirements
To run the code, make sure you have the following libraries installed:

numpy

torch

matplotlib

You can install the dependencies using:

pip install numpy torch matplotlib

How to Run the Code

1. Clone the repository
First, clone the repository to your local machine:

git clone https://github.com/PKSR-DS/Attention-Mechanisms-in-Transformers.git

cd Attention-Mechanisms-in-Transformers

2. Install dependencies
Make sure you have the necessary dependencies installed:

pip install numpy torch matplotlib


3. Run the script
You can run the attention_mechanisms.py script that contains the implementation of self-attention and multi-head attention. To run the script, use the following

command:
python attention_mechanisms.py

Expected Output
Running the script will display the following outputs:

Attention output and attention weights from the self-attention mechanism.

Attention Output:
 tensor([[0.8137, 0.4935, 0.5065, 0.1863],
        [0.4935, 0.8137, 0.1863, 0.5065],
        [0.7259, 0.7259, 0.2741, 0.2741]])
Attention Weights:
 tensor([[0.5065, 0.1863, 0.3072],
        [0.1863, 0.5065, 0.3072],
        [0.2741, 0.2741, 0.4519]])

A heatmap visualization of the attention weights.
![image](https://github.com/user-attachments/assets/48b2e59a-a930-4205-8f3d-4d42519059fd)


The output from the multi-head attention mechanism.
Output from Multi-Head Attention:
 tensor([[[ 0.1367, -0.2819,  0.0927,  0.4478],
         [-0.1354, -0.4186, -0.5878,  0.6200],
         [-0.2719, -0.4759, -0.4203,  0.1670]]], grad_fn=<ViewBackward0>)
