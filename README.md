
# Mixture of Experts From Scratch

This repository is my hands-on exploration of Mixture of Experts, starting from the fundamentals and gradually moving toward modern MoE architectures, routing, sparse computation, load balancing, training, distributed execution, inference, and production systems.

The main goal is to understand **why Mixture of Experts was introduced, what problem it solves, how it changes the Transformer architecture, and what actually happens to every token inside an MoE model**.

I do not want to treat MoE as simply another model architecture that can be enabled through a library.

I want to build the important components myself and understand the complete path:

```text
Dense Transformer
        ↓
Why Dense Scaling Becomes Expensive
        ↓
Why Sparse Computation Helps
        ↓
Mixture of Experts
        ↓
Experts
        ↓
Router
        ↓
Top-K Routing
        ↓
Token Dispatch
        ↓
Expert Computation
        ↓
Load Balancing
        ↓
MoE Training
        ↓
Distributed MoE
        ↓
Inference
        ↓
Production
```

---

# What Is Mixture of Experts?

Mixture of Experts, commonly called **MoE**, is a neural network architecture that increases the model's total parameter capacity while activating only a subset of those parameters for each input token.

The key idea is simple:

```text
Instead of using one large network for every token,

use many specialized networks called Experts

and use a Router to decide which experts should process each token.
```

A simplified MoE layer looks like:

```text
                         Input Token
                              |
                              v
                           Router
                              |
                    Routing Probabilities
                              |
                         Top-K Selection
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
           Expert 1        Expert 2        Expert N
              |               |               |
              +---------------+---------------+
                              |
                              v
                     Weighted Combination
                              |
                              v
                         MoE Output
```

The important idea is:

```text
Many Parameters
      ↓
Only Some Parameters Activated
      ↓
Sparse Computation
```

This creates a fundamental distinction:

```text
Total Model Parameters
            ≠
Active Parameters Per Token
```

For example, an MoE model could contain many experts but route each token to only two of them.

```text
Total Experts = 16
Top-K = 2

Token
  ↓
Router
  ↓
Expert 4 + Expert 11
```

The token does not need to pass through all 16 experts.

---

# Why Did MoE Come?

To understand MoE, it is important to first understand the scaling problem of traditional dense Transformers.

A standard Transformer contains layers such as:

```text
Input
  ↓
Self Attention
  ↓
Feed Forward Network
  ↓
Output
```

The Feed Forward Network, or FFN, contains a large number of parameters.

For every token, the same FFN is activated.

```text
Token 1 ──┐
Token 2 ──┤
Token 3 ──┼──→ Same FFN
Token 4 ──┤
Token 5 ──┘
```

As models become larger, increasing the size of this shared FFN increases the number of parameters and computation required for every token.

This creates a scaling tradeoff:

```text
Larger Dense Model
       ↓
More Parameters
       ↓
More Compute Per Token
       ↓
Higher Training Cost
       ↓
Higher Inference Cost
```

The natural question becomes:

```text
Can we increase the total capacity of the model
without using the entire model for every token?
```

This is where the Mixture of Experts approach becomes useful.

---

# Dense Transformer vs Mixture of Experts

The most important difference is how the feed-forward computation is performed.

## Traditional Dense Transformer

A simplified Transformer block looks like:

```text
Input
  |
  v
Self Attention
  |
  v
FFN
  |
  v
Output
```

The same FFN processes every token.

For example:

```text
Token A ──→ FFN
Token B ──→ FFN
Token C ──→ FFN
Token D ──→ FFN
```

There is no expert selection.

Every token uses the same feed-forward parameters.

---

## MoE Transformer

An MoE Transformer replaces the shared FFN with multiple expert FFNs and a router.

```text
Input
  |
  v
Self Attention
  |
  v
Router
  |
  +--------+--------+--------+
  |        |        |        |
  v        v        v        v
 Expert 1 Expert 2 Expert 3 ... Expert N
  |        |        |        |
  +--------+--------+--------+
           |
           v
        Combine
           |
           v
         Output
```

Now different tokens can be processed by different experts.

For example:

```text
Token A → Expert 2
Token B → Expert 7
Token C → Expert 2
Token D → Expert 11
```

The same model therefore has many possible computational paths.

---

# Dense vs MoE: The Core Difference

The difference can be summarized as:

```text
DENSE TRANSFORMER

Token
  ↓
Attention
  ↓
One Shared FFN
  ↓
Output
```

versus:

```text
MOE TRANSFORMER

Token
  ↓
Attention
  ↓
Router
  ↓
Select Experts
  ↓
Selected FFNs
  ↓
Combine
  ↓
Output
```

The dense model uses the same FFN for all tokens.

The MoE model introduces conditional computation.

```text
Dense:

Every Token
    ↓
Same Parameters


MoE:

Every Token
    ↓
Router
    ↓
Different Selected Parameters
```

This is the fundamental architectural difference.

---

# Why Not Just Make One Huge Dense Model?

Suppose we want a model with a very large number of parameters.

With a dense architecture:

```text
Very Large Model
       ↓
All Parameters Available
       ↓
All Relevant Layers Activated
       ↓
High Compute Per Token
```

With MoE:

```text
Very Large Model
       ↓
Many Experts
       ↓
Router
       ↓
Only Selected Experts Activated
       ↓
Lower Active Compute Per Token
```

This allows us to increase the model's total capacity without requiring every token to use every expert.

The important concept is therefore:

```text
Model Capacity
        ↓
Can Scale Up

Active Computation
        ↓
Can Remain Relatively Sparse
```

This is one of the main reasons MoE became an important approach for scaling neural networks.

---

# What Does "Expert" Actually Mean?

An expert is usually a separate neural network, commonly an FFN or MLP.

A simplified expert is:

```text
Expert

Input
  ↓
Linear Layer
  ↓
Activation
  ↓
Linear Layer
  ↓
Output
```

If we have four experts:

```text
             +---------- Expert 1
             |
             +---------- Expert 2
Input → Router
             +---------- Expert 3
             |
             +---------- Expert 4
```

Each expert has its own parameters.

For example:

```text
Expert 1 → W1
Expert 2 → W2
Expert 3 → W3
Expert 4 → W4
```

They are not simply copies of the same parameter tensor.

Each expert can learn different transformations during training.

---

# Does Each Expert Have a Manually Assigned Job?

No.

This is an important point.

We normally do not explicitly tell the model:

```text
Expert 1 = Mathematics
Expert 2 = Programming
Expert 3 = English
Expert 4 = Science
```

Instead, the router and experts are trained jointly.

Over training, different experts may develop different preferences or specialization patterns.

For example, analysis may reveal that certain experts are more frequently selected for:

```text
Code
Mathematics
Natural Language
Technical Text
Specific Linguistic Patterns
```

But this specialization is an emergent property, not necessarily a manually defined assignment.

This repository will investigate whether and how such specialization appears.

---

# The Router

The router is one of the most important components of an MoE model.

Its job is to determine:

```text
Which expert should process this token?
```

Given a token representation:

```text
x ∈ R^d
```

the router can produce scores for all experts.

For example:

```text
Token Representation
        ↓
Router
        ↓
Expert Scores

E1 → 0.10
E2 → 0.65
E3 → 0.05
E4 → 0.20
```

If the model uses Top-1 routing:

```text
E2 → Selected
```

If the model uses Top-2 routing:

```text
E2 → Selected
E4 → Selected
```

The router therefore determines the computational path taken by each token.

---

# Router Computation

A simplified router can be represented as:

```text
x
↓
Linear Projection
↓
Router Logits
↓
Softmax
↓
Routing Probabilities
```

Mathematically:

```text
r = xW_r
```

where:

```text
x  = token hidden representation
W_r = router parameters
r  = expert routing logits
```

The router then determines which experts should receive the token.

A simplified representation is:

```text
P(expert | token)
```

The highest scoring experts are selected according to the routing strategy.

---

# What Is Sparse Activation?

Sparse activation means that only a subset of the available expert parameters is activated for a particular token.

Suppose:

```text
Number of Experts = 16
Top-K = 2
```

A dense approach would conceptually use:

```text
16 / 16 Experts
```

while the MoE routing decision uses:

```text
2 / 16 Experts
```

for each token.

So:

```text
Total Expert Capacity
        ↓
16 Experts

Active Computation
        ↓
2 Experts Per Token
```

This is called sparse activation.

The model can therefore have a large total parameter count while maintaining a smaller active computation path.

---

# Total Parameters vs Active Parameters

This distinction is critical when discussing MoE models.

Consider:

```text
16 Experts
Each Expert = 500M Parameters
```

The expert parameters alone represent:

```text
16 × 500M
= 8B Parameters
```

But if:

```text
Top-K = 2
```

a token may only activate:

```text
2 × 500M
= 1B Expert Parameters
```

ignoring shared Transformer parameters for simplicity.

Therefore:

```text
Total Parameters
        ↓
8B Expert Parameters

Active Expert Parameters
        ↓
1B Per Token
```

This does not mean the entire model only contains 1B parameters.

It means only a subset is involved in processing that token.

---

# Conditional Computation

This is the deeper concept behind MoE.

A dense network follows approximately:

```text
Input
  ↓
Fixed Computational Path
  ↓
Output
```

An MoE network follows:

```text
Input
  ↓
Router
  ↓
Conditional Computational Path
  ↓
Selected Experts
  ↓
Output
```

Different inputs can therefore activate different parts of the model.

```text
Token A → Expert 2
Token B → Expert 5
Token C → Expert 2
Token D → Expert 9
```

The computation becomes input-dependent.

---

# Why This Is Useful

The major motivation is to separate:

```text
How much the model can represent
```

from:

```text
How much computation every token requires
```

In a dense model, increasing model capacity generally increases computation for every token.

In an MoE model, we can increase the number of experts while keeping the number of selected experts relatively small.

Conceptually:

```text
Dense Scaling

More Parameters
      ↓
More Active Parameters
      ↓
More Compute


MoE Scaling

More Total Parameters
      ↓
More Experts
      ↓
Same / Small Number of Selected Experts
      ↓
Sparse Active Compute
```

This makes MoE particularly interesting for large-scale model design.

---

# But MoE Is Not Free

MoE does not magically solve the scaling problem.

It introduces a new set of engineering challenges.

Instead of only worrying about:

```text
Compute
Memory
```

we now also need to care about:

```text
Routing
Expert Balance
Token Dispatch
Expert Capacity
Communication
All-to-All Operations
GPU Utilization
Training Stability
Inference Latency
```

For example, if the router sends too many tokens to one expert:

```text
Tokens
  ↓
Router
  ↓
Expert 1
  ↓
Too Many Tokens
  ↓
Overloaded
```

while other experts may remain underutilized:

```text
Expert 2 → Few Tokens
Expert 3 → Few Tokens
Expert 4 → Few Tokens
```

This creates an expert load-balancing problem.

---

# The Core MoE Problem

MoE therefore introduces a new optimization problem:

```text
How do we route tokens to a small number of experts
while keeping the experts useful, balanced, and efficient?
```

This leads to several important concepts:

```text
Router
   ↓
Top-K Routing
   ↓
Token Dispatch
   ↓
Expert Capacity
   ↓
Load Balancing
   ↓
Expert Utilization
   ↓
Sparse Computation
```

And when the experts are distributed across GPUs:

```text
Router
   ↓
Expert Assignment
   ↓
Token Communication
   ↓
Remote Expert
   ↓
Expert Computation
   ↓
Communication Back
```

This is where MoE becomes both a machine learning problem and a systems engineering problem.

---

# Where MoE Fits Inside a Transformer

MoE does not normally mean replacing the entire Transformer.

A common architecture replaces the FFN portion of a Transformer block with an MoE layer.

A simplified dense Transformer block:

```text
Input
  ↓
Self Attention
  ↓
FFN
  ↓
Output
```

can become:

```text
Input
  ↓
Self Attention
  ↓
MoE FFN
  ↓
Output
```

where:

```text
MoE FFN

       Input
          ↓
       Router
          ↓
      Top-K Experts
          ↓
   Expert Computation
          ↓
   Weighted Combination
          ↓
        Output
```

The attention mechanism can remain shared across tokens while the FFN computation becomes sparse and conditional.

---

# Dense Transformer vs MoE Transformer

| Property          | Dense Transformer                              | MoE Transformer                              |
| ----------------- | ---------------------------------------------- | -------------------------------------------- |
| FFN               | One shared FFN                                 | Multiple expert FFNs                         |
| Routing           | No                                             | Yes                                          |
| Expert Selection  | No                                             | Top-K / routing strategy                     |
| Active FFN        | Same FFN for every token                       | Selected experts                             |
| Total Parameters  | Usually smaller for a comparable sparse design | Can be much larger                           |
| Active Parameters | Closely tied to total model size               | Smaller than total parameters                |
| Computation       | Dense                                          | Sparse in expert layer                       |
| Load Balancing    | Not an expert-routing issue                    | Major concern                                |
| Token Dispatch    | Not required                                   | Required                                     |
| Communication     | Standard distributed communication             | Can require expert communication             |
| Implementation    | Simpler                                        | More complex                                 |
| Scaling           | Compute grows with model size                  | Capacity can grow faster than active compute |

The important point is not that MoE is simply "better" than a dense Transformer.

The real question is:

```text
For a particular scale and hardware environment,

does the additional model capacity justify

the routing, communication, memory, and systems complexity?
```

That tradeoff is something I want to investigate experimentally in this repository.

---

# When Did MoE Come From?

The idea of Mixture of Experts is much older than modern LLMs.

Mixture of Experts originated as a general machine learning architecture based on combining multiple specialized models through a gating or routing mechanism.

The central idea was already:

```text
Input
  ↓
Gating / Routing
  ↓
Select Specialized Models
  ↓
Combine Their Outputs
```

The concept later evolved through different neural network architectures and became particularly important again with the rise of large-scale deep learning.

Modern Transformer-based MoE systems brought the idea into large language models by replacing dense feed-forward layers with collections of experts and using learned routing to activate only a subset of them.

So the progression can be viewed conceptually as:

```text
Classical Mixture of Experts
          ↓
Neural Network MoE
          ↓
Sparse MoE
          ↓
Transformer MoE
          ↓
Large-Scale Language Model MoE
          ↓
Distributed MoE Systems
```

The important development was not simply adding more networks.

It was combining:

```text
Large Parameter Capacity
        +
Sparse Activation
        +
Learned Routing
        +
Distributed Training
```

---

# Why Modern LLMs Use MoE

As language models became larger, simply scaling dense architectures became increasingly expensive.

The model needs enough capacity to represent complex patterns across:

```text
Language
Code
Mathematics
Reasoning
Knowledge
Different Domains
Different Languages
```

A dense model processes every token through the same FFN parameters.

MoE provides another scaling strategy:

```text
Many Experts
     ↓
Large Total Capacity
     ↓
Router
     ↓
Small Number of Active Experts
     ↓
Sparse Computation
```

This allows researchers and engineers to investigate a different point in the:

```text
Quality
   ↕
Compute
   ↕
Memory
   ↕
Communication
   ↕
Cost
```

tradeoff space.

---

# What I Want to Understand

This repository is not only about implementing a router.

I want to understand the entire MoE system.

```text
Why was MoE introduced?

What problem does it solve?

How is MoE different from a dense Transformer?

Where exactly is MoE inserted into a Transformer?

What is an expert?

How does a router make decisions?

What does Top-K routing mean?

Why is sparse activation useful?

How are tokens dispatched?

How are expert outputs combined?

What happens when an expert receives too many tokens?

Why does load balancing matter?

What is routing collapse?

Can experts actually specialize?

How does MoE affect parameter count?

How many parameters are active per token?

How does MoE affect FLOPs?

What happens during backpropagation?

How does distributed expert parallelism work?

Why is All-to-All communication required?

When does communication become the bottleneck?

Is MoE actually faster than a dense model?

How does MoE affect inference latency?

How does MoE scale across GPUs?

How should MoE models be evaluated?

What are the real production tradeoffs?
```

---

# Learning Direction

The repository will move from a very small implementation toward increasingly realistic systems.

```text
Dense FFN
   ↓
Simple MoE
   ↓
Multiple Experts
   ↓
Router
   ↓
Top-1 Routing
   ↓
Top-2 Routing
   ↓
Top-K Routing
   ↓
Token Dispatch
   ↓
Expert Combination
   ↓
Sparse Activation
   ↓
Capacity Management
   ↓
Load Balancing
   ↓
Expert Specialization
   ↓
MoE Transformer
   ↓
MoE Training
   ↓
Distributed MoE
   ↓
Expert Parallelism
   ↓
All-to-All Communication
   ↓
MoE Inference
   ↓
Optimization
   ↓
Production Analysis
```

The rest of this repository builds each of these components step by step, with implementations, tensor shapes, experiments, measurements, and comparisons against dense Transformer architectures.

---

# Complete MoE Flow

The core flow I will implement and investigate is:

```text
Input Tokens
      ↓
Token Embeddings
      ↓
Transformer Attention
      ↓
Token Hidden States
      ↓
Router
      ↓
Routing Logits
      ↓
Routing Probabilities
      ↓
Top-K Expert Selection
      ↓
Token-to-Expert Assignment
      ↓
Token Dispatch
      ↓
Expert Networks
      ↓
Expert Outputs
      ↓
Weighted Combination
      ↓
Restore Token Ordering
      ↓
MoE Output
      ↓
Next Transformer Layer
```

The deeper systems flow is:

```text
                         MOE TRANSFORMER

Input Tokens
      ↓
Embeddings
      ↓
Attention
      ↓
MoE Layer
      ↓
Router
      ↓
Top-K Selection
      ↓
Token Dispatch
      ↓
+---------+---------+---------+
|         |         |         |
E1        E2        E3       EN
|         |         |         |
+---------+---------+---------+
          ↓
   Expert Computation
          ↓
   Weighted Combination
          ↓
     MoE Output
          ↓
     Next Layer
```

This is the foundation for the remaining sections of the repository.
