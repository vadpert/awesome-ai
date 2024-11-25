# awesome-ai [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> Carefully curated list of AI research

## Contents
- 

## People
Noam Brown
Ilge Akkaya
Hunter Lightman

https://x.com/aimodelsfyi

-----------------------------
## Generative AI

###
- https://press.airstreet.com/
- https://paperswithcode.com/sota
- https://research.google/blog


### Platform Experience
- https://www.anthropic.com/news/analysis-tool


### Overview
[State of AI Report - 2024 ONLINE](https://docs.google.com/presentation/d/1GmZmoWOa2O92BPrncRcTKa15xvQGhq7g4I4hJSNlC0M/edit#slide=id.g3058058dd40_3_39)

### Prompt Engineering
- https://www.reddit.com/r/ClaudeAI/comments/1e39tvj/sonnet_35_coding_system_prompt_v2_with_explainer/
- https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags

- 

### Transformers
- https://towardsdatascience.com/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876

### LLM Papers
- Llama3 Herd of Models - https://arxiv.org/pdf/2407.21783
- 

### RAG
- Anthropic solved this using ‘contextual embeddings’, where a prompt instructs the model to generate text explaining the context of each chunk in the document
- prompt https://github.com/mlbrnm/contextualretrieval/blob/main/contextgeneration.py
- https://www.anthropic.com/news/contextual-retrieval

## Frameworks
- Optimizing inference proxy for LLMs - https://github.com/codelion/optillm


## Techniques
1. CoT with Reflection		-	Implements chain-of-thought reasoning with <thinking>, <reflection> and <output> sections
1. PlanSearch		-	Implements a search algorithm over candidate plans for solving a problem in natural language
1. ReRead		-	Implements rereading to improve reasoning by processing queries twice
1. Self-Consistency		-	Implements an advanced self-consistency method
1. Z3 Solver		-	Utilizes the Z3 theorem prover for logical reasoning
1. R* Algorithm		-	Implements the R* algorithm for problem-solving
1. LEAP		-	Learns task-specific principles from few shot examples
1. Round Trip Optimization		-	Optimizes responses through a round-trip process
1. Best of N Sampling		-	Generates multiple responses and selects the best one
1. Mixture of Agents		-	Combines responses from multiple critiques
1. Monte Carlo Tree Search		-	Uses MCTS for decision-making in chat responses
1. PV Game		-	Applies a prover-verifier game approach at inference time
1. CoT Decoding		- proxy	Implements chain-of-thought decoding to elicit reasoning without explicit prompting
1. Entropy Decoding	-	Implements adaptive sampling based on the uncertainty of tokens during generation

## Features
1. Router		-	Uses the optillm-bert-uncased model to route requests to different approaches based on the user prompt
1. Chain-of-Code		-	Implements a chain of code approach that combines CoT with code execution and LLM based code simulation
   1. Chain of Code: Reasoning with a Language Model-Augmented Code Emulator - https://arxiv.org/pdf/2312.04474
1. Memory		-	Implements a short term memory layer, enables you to use unbounded context length with any LLM
1. Privacy		-	Anonymize PII data in request and deanonymize it back to original value in response
1. Read URLs		-	Reads all URLs found in the request, fetches the content at the URL and adds it to the context
1. Execute Code		-	Enables use of code interpreter to execute python code in requests and LLM generated responses



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


-------------------------------------------
### Metrics
- https://medium.com/data-science-at-microsoft/evaluating-llm-systems-metrics-challenges-and-best-practices-664ac25be7e5



-------------------------------------------
### OCR
- https://microsoft.github.io/OmniParser - screen parsing
- https://yolov8.com/ - object detection / vision model


-------------------------------------------
## Distance / Text Matching
- Levenshetin Distance - measure the edit distance between strings
- Jaccard Similarity - compare the overlap of characters and tokens
- Cosine Similarity - convert names to vectors and measure cosine distance of vectors

- Named Entity Recognition (NER)
- FuzzyWuzzy / RapidFuzz
- Word2Vec, GloVe, FastText
- Latent Dirichlet Allocation (LDA)
- Non-Negative Matrix Factorization (NMF)
- Siamese Neural Networks
- Locality Sensitive Hashing (LSH) - datasketch / falconn

- Faiss (Facebook AI Similarity Search)
- Annoy (Approximate Nearest Neighbors Oh Yeah)
- NMSLIB (Non-Metric Space Library)


-------------------------------------------
## Reinforcement Learning
1. Imitation Learning (IL)
1. (Reinforcement Learning: An Introduction 2018) [https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf]
1. deep reinforcement learning (RL)
1. Monte Carlo tree search
1.  Contextual Bandits (one step RL)
1.  Value Based
1. Policy Based
1.  Actor/Critic
1.  Model Based
1.  Transformer Based



-------------------------------------------
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



## Process / Workflow
- (UX) https://excalidraw.com/
- (Amazon State Language) https://states-language.net/
- Function as a Service (FaaS) paradigm


### Enterprise Automation (RPA) Robotic Process Automation
- https://arxiv.org/abs/2405.03710 (ECLAIR) - ECLAIR
- https://arxiv.org/abs/2404.13050 (FlowMind) - JPM
- 



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
2. Microsoft Magentic One - https://github.com/microsoft/autogen/blob/main/python/packages/autogen-magentic-one/src/autogen_magentic_one/agents/orchestrator_prompts.py

## Video / Image Generation / Art
1. https://civitai.com/tag/base%20model
1. https://huggingface.co/enhanceaiteam/
1. Flux Model - https://blackforestlabs.io/flux-1/


## Rough Path Signatures


# Graph
1. Leiden Algorith

2. Laplacian / Heat Diffusion Graphs



