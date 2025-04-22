# Content Generation LLM

This repository presents an end-to-end pipeline for **language model-based content generation**, integrating traditional NLP techniques with modern deep learning architectures. The project is structured to incrementally enhance text generation — starting from lexical and semantic augmentation, progressing to statistical and neural generation mechanisms.

---

## Project Overview

The goal of this project is to generate **coherent, fluent, and contextually relevant text** from user-provided prompts by leveraging a combination of:

- Lexical and semantic representation techniques (TF-IDF, BM25, Word2Vec, GloVe)
- Statistical models (Trigram, 4-Gram)
- Deep learning-based language models (Transformer, Reformer, fine-tuned GPT-2)

These techniques are not isolated but integrated into a **hybrid generation pipeline** that enhances intermediate prompts with keyword expansion before feeding them into downstream generators.

---

## Dataset

**BookCorpus**  
We use the [BookCorpus Dataset](https://huggingface.co/datasets/bookcorpus), a large, diverse corpus of text from over 11,000 books. Its natural language structure makes it ideal for training language models capable of generating fluent long-form text.

---

## Models and Techniques

### Text Representation and Prompt Enrichment
- **TF-IDF & BM25**: Used for lexical keyword extraction to enrich prompt content.
- **Word2Vec & GloVe**: Used for semantic vector augmentation of prompts, enabling context expansion beyond exact token matches.

### Probabilistic Generation
- **Trigram and 4-Gram Models**:
  - Trained on the tokenized corpus using count-based probability estimates.
  - Enhanced with:
    - Memory buffer to reduce repetition
    - Sampling from word distributions (not greedy)
    - Duplicate phrase removal

### Neural Generation Models

- **Transformer (TensorFlow/Keras)**:
  - Custom decoder-only model using TensorFlow/Keras for structured sequence generation.
  - Trained with categorical cross-entropy on padded sequences.

- **Reformer (PyTorch)**:
  - Optimized for long sequences using:
    - LSH attention (sub-quadratic complexity)
    - Greedy decoding for faster inference
    - AMP (Automatic Mixed Precision) for resource-efficient training

- **GPT-2 (HuggingFace Transformers)**:
  - Fine-tuned on our processed dataset
  - Used top-k and top-p (nucleus) sampling
  - Tuned for prompt-based creative generation

---

## 📈 Evaluation Metrics

Text outputs were assessed through both automated and human evaluation:

- **BLEU Score**: Quantitative comparison with reference text
- **Perplexity**: Model confidence in next-word prediction
- **Human Evaluation**: Ratings based on:
  - Grammatical fluency
  - Semantic coherence
  - Relevance to input prompt

---

## Results Summary

| Component        | Purpose                             | Strengths                         |
|------------------|-------------------------------------|-----------------------------------|
| TF-IDF, BM25     | Lexical expansion                   | Fast, interpretable               |
| Word2Vec, GloVe  | Semantic augmentation               | Context-aware, generalizable      |
| N-Gram Models    | Statistical generation              | Lightweight, interpretable        |
| Transformer      | Structured neural generation        | Long-term dependencies            |
| Reformer         | Long-form generation (efficient)    | Memory-optimized, scalable        |
| GPT-2 (Fine-tuned)| Coherent, fluent creative output    | High-quality, prompt-conditioned  |

---

## Future Work

- Integrate **RLHF** (Reinforcement Learning from Human Feedback) for goal-driven generation
- Implement a **web interface** for real-time generation with user prompts
- Add **BERTScore** or cosine similarity-based evaluators for semantic assessment
- Extend to **multilingual generation** using mT5 or mBART
- Explore **multimodal generation** using cross-modal embeddings (text + image/audio)

---

## Contributors

- **Madhesh Hariharran S**  
- Kavin Akash  
- Latchumi Raman R

> For full methodology and experimental analysis, see [`Content Generation LLM report.pdf`](./Content%20Generation%20LLM%20report.pdf)

---

## License

This project is licensed under the [MIT License](./LICENSE).


