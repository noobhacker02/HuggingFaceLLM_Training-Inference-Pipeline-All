# HuggingFace LLM Training & Inference Pipeline

This repository contains multiple notebooks and scripts demonstrating end-to-end workflows for **Large Language Models (LLMs)** and NLP pipelines using **Hugging Face**. It includes training, inference, question answering, sentiment analysis, NER, and more.

---

## Repository Structure


├── AutoModelForQuestionAnswering.ipynb <br>
├── CausalLLM_All.ipynb<br>
├── Classification_Sentiment_Analysis_Custom_Infer_Pipeline_HuggingFace_all_in_one.ipynb<br>
├── NERLLM_train_infer_pipeline.ipynb<br>
├── PipeLine_NLP_All.ipynb<br>
├── Seq2SeqLM.ipynb<br>
├── notebook_dir.py<br>
└── README.md<br>



---

## Notebooks Overview

### 1. `AutoModelForQuestionAnswering.ipynb`
- Implements **Question Answering (QA)** pipelines using Hugging Face models.
- Demonstrates fine-tuning, inference, and evaluation of QA models.

### 2. `CausalLLM_All.ipynb`
- Covers **causal language models** for text generation.
- Includes training and inference workflows for autoregressive LLMs.

### 3. `Classification_Sentiment_Analysis_Custom_Infer_Pipeline_HuggingFace_all_in_one.ipynb`
- Complete **sentiment analysis pipeline** with custom inference.
- Demonstrates preprocessing, model prediction, and evaluation of text classification tasks.

### 4. `NERLLM_train_infer_pipeline.ipynb`
- **Named Entity Recognition (NER)** pipeline for entity extraction from text.
- Includes training and inference using Hugging Face token classification models.

### 5. `PipeLine_NLP_All.ipynb`
- A **general NLP pipeline** notebook covering multiple tasks like text classification, NER, QA, summarization, etc.
- Serves as a unified pipeline for quick experimentation.

### 6. `Seq2SeqLM.ipynb`
- Implements **Sequence-to-Sequence models** for tasks like translation or summarization.
- Covers training and inference using Hugging Face `Seq2Seq` models.

### 7. `notebook_dir.py`
- Python utility to **organize, load, and run notebooks programmatically**.
- Helps in managing notebook execution or integration with pipelines.

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/noobhacker02/HuggingFaceLLM_Training-Inference-Pipeline-All.git
cd HuggingFaceLLM_Training-Inference-Pipeline-All

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
````

> ⚠️ Ensure you have `transformers`, `datasets`, `torch`, and `accelerate` installed. Some notebooks may require additional packages.

---

## How to Use

1. Open the desired notebook in **Jupyter Notebook / Jupyter Lab / VSCode**.
2. Follow the cells sequentially for training or inference.
3. You can customize datasets and models according to your needs.
4. Use `notebook_dir.py` to programmatically run or manage notebooks if required.

---

## Contribution

Contributions are welcome!
Feel free to fork this repository, add features, improve pipelines, or fix issues.

---

## License

This repository is open-source. Use it under the terms of the MIT License.

```

