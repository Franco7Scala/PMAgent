# Beyond Dashboards: PMAgent for RAG-Driven Natural Language Process Mining

[![Paper](https://img.shields.io/badge/Paper-ECML_PKDD-brightgreen.svg)](TODO)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and resources for the **PMAgent** framework presented in the research paper "Beyond Dashboards: PMAgent for RAG-Driven Natural Language Process Mining". Our proposal addresses the accessibility barrier in Process Mining (PM) by integrating Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG). This integration enables users to execute advanced analytical tasks, such as predictive monitoring and what-if analysis, entirely through natural language. 

## Key Features

* **Trace-to-Text Translation:** Ingests standard tabular event logs and translates them into semantic trace histories with dynamic metadata serialization. This transformation allows the framework to represent complex process behaviors and sequential dependencies in a format natively understandable by language models;
* **RAG-Driven Architecture:** Grounds the LLM's reasoning in retrieved, fact-based process contexts rather than relying solely on parametric memory. It utilizes a high-performance vector database (FAISS) and text embedding models to minimize hallucinations and ensure reliable insights;
* **Agentic Decision Support:** Operates as a multi-task agent capable of zero-shot reasoning for next-step prediction, scenario evaluation, and adversarial query rejection. It manages persistent state and multi-step analytical reasoning without requiring task-specific retraining;
* **Green AI Alignment:** Bypasses energy-intensive training and fine-tuning loops by pairing a completely frozen pre-trained LLM (Google Gemma-3) with explicit non-parametric memory.

## Repository Structure

* **`src/`**: Core source code directory.
  * `main.py` / `pm_agent.py`: Main entry points to initialize the LLM orchestrator and execute conversational queries;
  * `preprocessing/`: Contains the Data Preprocessing module for the Trace-to-Text translation of raw event logs;
  * `rag/`: Implements the Retrieval-Augmented Generation pipeline, including character-based text splitters and FAISS vector database integration;
  * `llm/`: Contains the LangChain integration and the strict system prompt templates used to enforce analytical boundaries and mitigate hallucinations;
  * `datasets/`: Directory intended for event logs, such as the standard real-world Helpdesk dataset used for evaluation.
    
* **`requirements.txt`**: List of project dependencies.

## Requirements

To run the code, ensure you have the libraries specified in `requirements.txt` installed. Based on the framework's architecture, the main dependencies include:

* Python (>=3.8)
* LangChain
* FAISS
* Transformers (Hugging Face)
* Pandas (for tabular data manipulation)

## Citation

```bibtex
Coming soon...
