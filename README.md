# awesome-ai [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> Carefully curated list of AI research

## Contents
- 

## People
Noam Brown
Ilge Akkaya
Hunter Lightman

https://x.com/aimodelsfyi


## History of AI
- https://en.wikipedia.org/wiki/Alonzo_Church
- Church–Turing thesis - https://plato.stanford.edu/entries/church-turing/
- https://en.wikipedia.org/wiki/Halting_problem
-  Entscheidungsproblem ("decision problem")
-  Newell, Allen, and Herbert A. Simon, Human Problem Solving, Prentice-Hall, Englewood Cliffs, NJ., 1972.




## Cognitive Architecture
-  Cognitive Architectures for Language Agents - https://arxiv.org/pdf/2309.02427
- Characterizing Technical Debt and Antipatterns in AI-Based Systems: A Systematic Mapping Study - https://arxiv.org/pdf/2103.09783

- Cognitive Architecture
- structured action space
- language-based general intelligence
- control flows
- grounding to existing knowledge or external observations
- cognitive language agents
- (memory, action, and decision-making),
   - information storage (divided into working and long-term memories)
   - their action space (divided into internal and external actions)
   - decision-making procedure (which is structured as an interactive loop with planning and execution)
   - information retrieval: rule-based, sparse, or dense retrieval
-  Generative Agents
   - (Park et al., 2023) are language agents grounded to a sandbox game affording interaction
 with the environment and other agents. Its action space also has all four kinds of actions: grounding, reasoning,
 retrieval, and learning. Each agent has a long-term episodic memory that stores events in a list. These agents
 use retrieval and reasoning to generate reflections on their episodic memory (e.g., “I like to ski now.”) which
 are then written to long-term semantic memory. During decision-making, it retrieves relevant reflections from
 semantic memory, then reasons to make a high-level plan of the day. While executing the plan, the agent
 receives a stream of grounding observations; it can reason over these to maintain or adjust the plan.
   - recon learn episodic and long term memory how to solve specific accounts
- Standard Interaction
   - agents should be structured and modular
   -  Memory, Action, Agent
   - maintaining a single company-wide “language agent library” would reduce technical debt (Sculley et al., 2014; Lwakatare et al., 2020) by facilitating testing and component re-use across individual agent deployments.
   -  adaptively allocate computation (Russek et al., 2022; Lieder and Griffiths, 2020; Callaway et al., 2022; Gershman et al., 2015).
   -  

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
- https://learnprompting.org/docs/introduction
- Prompt Ensembling - https://learnprompting.org/docs/reliability/ensembling
- Writer Prompting Strategies - https://dev.writer.com/home/prompting


| Type | Notes |
|-|-|
|Self-Critique|  |
|Socratic Models|  |
|Ask Me Anything (AMA) Prompting| | 


### Retrieval Techniques

| Type | Notes |
|-|-|
|Retrieval Distillion| |


### Reasoning Techniques
- classical planning algorithms:
- Tree of Thoughts (Yao et al., 2023)
- RAP (Hao et al., 2023)
- Monte Carlo Tree Search (MCTS; Browne et al., 2012)
- Breadth First Search | Depth First Search
- implementing tree search to mitigate myopia induced by autoregressive generation (Yao et al., 2023; Hao et al., 2023).


#### Legal
- https://github.com/HazyResearch/legalbench/blob/main/tasks/definition_extraction/README.md
- Contract Understanding Atticus Dataset (CUAD) - https://www.atticusprojectai.org/cuad


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
1. CoT with Reflection		-	Implements chain-of-thought reasoning with < thinking >, < reflection > and < output > sections
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


- [Chain of Code: Reasoning with a Language Model-Augmented Code Emulator](https://arxiv.org/abs/2312.04474) - [Implementation](https://github.com/codelion/optillm/blob/main/optillm/plugins/coc_plugin.py)
- [Entropy Based Sampling and Parallel CoT Decoding](https://github.com/xjdr-alt/entropix) - [Implementation](https://github.com/codelion/optillm/blob/main/optillm/entropy_decoding.py)
- [Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation](https://arxiv.org/abs/2409.12941) - [Evaluation script](https://github.com/codelion/optillm/blob/main/scripts/eval_frames_benchmark.py)
- [Writing in the Margins: Better Inference Pattern for Long Context Retrieval](https://www.arxiv.org/abs/2408.14906) - [Inspired the implementation of the memory plugin](https://github.com/codelion/optillm/blob/main/optillm/plugins/memory_plugin.py)
- [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200) - [Implementation](https://github.com/codelion/optillm/blob/main/optillm/cot_decoding.py)
- [Re-Reading Improves Reasoning in Large Language Models](https://arxiv.org/abs/2309.06275) - [Implementation](https://github.com/codelion/optillm/blob/main/optillm/reread.py)
- [In-Context Principle Learning from Mistakes](https://arxiv.org/abs/2402.05403) - [Implementation](https://github.com/codelion/optillm/blob/main/optillm/leap.py)
- [Planning In Natural Language Improves LLM Search For Code Generation](https://arxiv.org/abs/2409.03733) - [Implementation](https://github.com/codelion/optillm/blob/main/optillm/plansearch.py)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) - [Implementation](https://github.com/codelion/optillm/blob/main/optillm/self_consistency.py)
- [Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers](https://arxiv.org/abs/2408.06195) - [Implementation](https://github.com/codelion/optillm/blob/main/optillm/rstar.py)
- [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692) - [Inspired the implementation of moa](https://github.com/codelion/optillm/blob/main/optillm/moa.py)
- [Prover-Verifier Games improve legibility of LLM outputs](https://arxiv.org/abs/2407.13692) - [Implementation](https://github.com/codelion/optillm/blob/main/optillm/pvg.py)
- [Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning](https://arxiv.org/abs/2405.00451) - [Inspired the implementation of mcts](https://github.com/codelion/optillm/blob/main/optillm/mcts.py)
- [Unsupervised Evaluation of Code LLMs with Round-Trip Correctness](https://arxiv.org/abs/2402.08699) - [Inspired the implementation of rto](https://github.com/codelion/optillm/blob/main/optillm/rto.py)
- [Patched MOA: optimizing inference for diverse software development tasks](https://arxiv.org/abs/2407.18521) - [Implementation](https://github.com/codelion/optillm/blob/main/optillm/moa.py)
- [Patched RTC: evaluating LLMs for diverse software development tasks](https://arxiv.org/abs/2407.16557) - [Implementation](https://github.com/codelion/optillm/blob/main/optillm/rto.py)





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

- http://incompleteideas.net/book/RLbook2020.pdf
- https://huggingface.co/learn/deep-rl-course/en/unit0/introduction



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


# Robotics
- https://groups.google.com/g/rssc-list
- 


