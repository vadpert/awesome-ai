# awesome-ai [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> Carefully curated list of AI research

## Contents
- [Reinforcement Learning](#Reinforcement-Learning)
  - [Reinforcement Learning](#Reinforcement-Learning)
  - [Reinforcement Learning](#Reinforcement-Learning)
- [Reinforcement Learning](#Reinforcement-Learning)
- 

## People
Noam Brown
Ilge Akkaya
Hunter Lightman

https://x.com/aimodelsfyi?ref_src=aimodelsfyi

-----------------------------
## Generative AI

###
- https://press.airstreet.com/
- https://paperswithcode.com/sota
- https://research.google/blog



### Overview
[State of AI Report - 2024 ONLINE](https://docs.google.com/presentation/d/1GmZmoWOa2O92BPrncRcTKa15xvQGhq7g4I4hJSNlC0M/edit#slide=id.g3058058dd40_3_39)


### Transformers
- https://towardsdatascience.com/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876

### RAG
- Anthropic solved this using ‘contextual embeddings’, where a prompt instructs the model to generate text explaining the context of each chunk in the document
- prompt https://github.com/mlbrnm/contextualretrieval/blob/main/contextgeneration.py
- 

### Notes
- Genie (winner of a Best Paper award at ICML 2024)
- latent action space
- OpenDevin (code) generation
MultiOn is also betting big on RL, with its autonomous web agent - Agent Q (see slide 65) - combining search, self-critique, and RL
Meta’s TestGen-LLM has gone from paper to product at breakneck space (4 months), being integrated into Qodo’s Cover-Agent
- Haize Labs worked with Hugging Face to create the first ever red teaming resistance benchmark
-- Source: https://arxiv.org/abs/2402.04792 ( Google DeepMind team has combined the simplicity of direct alignment from preferences (DAP) with the on-line policy learning of RLHF to create direct alignment from AI feedback)
- Source: https://arxiv.org/abs/2406.08391 (uncertainty score)

-  Anthropic’s interpretability team used sparse autoencoders - neural networks that learn efficient representations of data by emphasizing important features and ensuring only a few are active at any one time - to decompose activations of Claude 3 Sonnet into interpretable components. They also showed that by ‘pinning’ a feature to ‘active’ you could control the output - famously turning up the strength of the Golden Gate feature.
- Source: https://www.anthropic.com/news/mapping-mind-language-model 
- Source: https://www.anthropic.com/news/golden-gate-claude
- Source: https://arxiv.org/abs/2406.04093 
Source: https://arxiv.org/abs/2403.03867 (Origins of Linear Representations)
Source: https://arxiv.org/abs/2405.12250 (Transformer is secretly linear)

Google has introduced a popular new method for decoding intermediate neurons. Patchscopes takes a hidden representation for LLM and ‘patching’ it to a different prompt. This prompt is used to generate a description or answer a question, revealing the encoded information.






### Enterprise Automation (RPA) Robotic Process Automation
- https://arxiv.org/abs/2405.03710 (ECLAIR) - ECLAIR
- https://arxiv.org/abs/2404.13050 (FlowMind) - JPM
- 

### Workflow 
- https://excalidraw.com/

- 

### Metrics
- https://medium.com/data-science-at-microsoft/evaluating-llm-systems-metrics-challenges-and-best-practices-664ac25be7e5


## Image Generation
- https://blackforestlabs.io/flux-1/

- 

-----------------------------
## Imitation Learning (IL)

-----------------------------

## Reinforcement Learning (RL)

(Reinforcement Learning: An Introduction 2018) [https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf]

deep reinforcement learning (RL)
Monte Carlo tree search

### Contextual Bandits (one step RL)


### Value Based

### Policy Based

### Actor/Critic

### Model Based

### Transformer Based

### Other




-----------------------------
## Traditional AI

### Regression (continous variables)
1. Linear Regression

### Classification (discrete variables)
1. *** Decision Trees
1. *** Random Forest
1. *** XGBoost - Gradient Boosting
2. *** LightGBM - Gradient Boosting
3. *** Support Vector Machines (SVMs)
 

### Binary Classification

1. *** Logistic Regression
2. 

### Similarity/Duplicate Detection Algorithms
1. *** Siamese Networks
2. *** K-Nearest Neighbors (KNN)

### Clustering

1. *** K-Means Clustering
3. *** DBSCAN

### Matrix Factorization

1. Collaborative Filtering

### Dimensionality Reduction

1. Principal Component Analysis (PCA)
1. 

### Neural Networks / Deep Learning Models

1. *** Autoencoders
1. *** Recurrent Neural Networks (RNNs)
1. *** Transformers
1. Deep Belief Network
1. Restricted Boltzmann Machine (RBM)
1. Hierarchical Temporal Memory (HTM)
1. Convolutional Neural Networks (CNN)
1. Long Short Term Memory (LSTM)

-----------------------------

## Probability Theory
- [Multi-armed bandit](http://www.com/) - Official website.



# Document Parsing
1. https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/partition/docx.py
2. 

# Legal entities
1. https://www.gleif.org/en/about-lei/introducing-the-legal-entity-identifier-lei

# Location
1. https://github.com/openvenues/libpostal


# Agents
1. CrewAI, Autogen, LangGraph, LlamaIndex Workflows, OpenAI Swarm, Vectara Agentic, Phi Agents, Haystack Agents
https://aiagentsdirectory.com/landscape
2. 

## Video / Image Generation / Art
1. https://civitai.com/tag/base%20model
2. Flux
3. https://huggingface.co/enhanceaiteam/


## Rough Path Signatures
1. 

