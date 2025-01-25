# AI Engineer Notes

## Overview

**AI**: Simulates human intelligence (e.g., NLP, computer vision, robotics).

**ML**: AI subset that learns from data:

- Supervised: Labeled data (e.g., spam detection).
- Unsupervised: Unlabeled data (e.g., clustering).

**DL**: ML subset with deep neural networks, often a "black box."

---

Data processing includes cleaning, normalization, and preprocessing via ETL pipelines.

Problems are regression or classification, solved using ML or DL algorithms.

Transformers (e.g., GPT) excel at language tasks but pose risks like evasion or poisoning; data anonymity is crucial.

---

**Model Creation**:

- **From Scratch**: Design architecture, optimize layers, train iteratively; resource-intensive.
- **Transfer Learning**: Adapt pre-trained models with fine-tuning; efficient for specific tasks.

**Steps**:

1. Define scope.
2. Align AI with business goals.
3. Consult stakeholders for KPIs.

---

### Common Terminology

- **ML**: Build models from scratch by designing architectures and training them.

- **AI Engineer**: Leverage pre-trained models for efficient implementation.

- **AI**: Mimics human intelligence, e.g., pattern recognition.

- **AGI (Artificial General Intelligence)**: Mimics human reasoning and task execution across diverse areas.

- **LLM (Large Language Model)**: Transformer-based models trained on extensive data, excelling at language tasks.

- **Inference**: Process where a trained ML model makes predictions on unseen data, e.g., a self-driving car identifying a new stop sign.

- **Training**: Teaching ML models to recognize patterns by exposing them to datasets. Models minimize errors by comparing predictions to correct answers and refining parameters using techniques like gradient descent.

---

### Key Concepts Simplified

- **Embeddings**: Dense vector representations of data; similar items are close in vector space. They simplify complex data and help models understand synonyms and antonyms by spatial closeness.

- **Vector Database**: Stores and retrieves high-dimensional vectors for fast similarity searches. Uses indexing techniques like Approximate Nearest Neighbor (ANN) for efficiency.

- **AI Agents**: Systems that interact with users, external environments, or other agents to perform complex tasks.

- **LangChain**: Framework for orchestrating LLMs with prompts, chains (sequences of functions), and memory. It integrates tools like vector databases and adds conversational memory.

- **TF-IDF**: Highlights important words in a text by measuring their frequency relative to normal usage (e.g., "coffee" has high value, but "and" is low).

- **Word2Vec**: Converts words into vectors using techniques like CBOW (predicts a word from its context) and Skip-gram (predicts context from a word).

- **GloVe**: Creates global vectors for word representation, emphasizing co-occurrence statistics.

- **Contextual Embeddings**: Dynamic representations of words based on their context, used in modern LLMs.

- **RAG (Retrieval Augmented** **Generation**): Combines data retrieval with LLMs to generate responses using up-to-date, relevant information.

- **Prompt Engineering:** Guide AI models, like GPT, to generate desired outputs.

