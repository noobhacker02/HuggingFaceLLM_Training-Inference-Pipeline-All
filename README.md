


# 🤖 HuggingFace LLM Training, Inference & Environment Setup

Welcome to **HuggingFaceLLM_Training-Inference-Pipeline-All** — a complete resource for experimenting with **Large Language Models (LLMs)** and other deep learning models using the Hugging Face ecosystem.  

This repository brings together:
- End-to-end **training & inference pipelines** for multiple NLP tasks  
- Organized **Jupyter notebooks** (NLP, Vision, Audio)  
- **Environment setup scripts** for easy reproducibility  
- Hands-on experiments with **Wav2Vec2 (Audio)** and **ViT (Vision)**  

---

## 📂 Repository Structure



.<br>
├── All-Collab-LLM's/           # Google Colab-friendly notebooks for LLM tasks <br>
│   └── README.md<br>
├── All-Jupyter-LLM's/          # Local Jupyter notebooks for training/inference<br>
│   └── README.md<br>
├── LLM Environment Setup/      # PowerShell script for automatic environment creation<br>
│   └── setup-llm.ps1<br>
├── LICENSE<br>
└── README.md                   # Main documentation (this file)<br>





## 📒 Notebooks Included

### 🔤 NLP Pipelines
- **AutoModelForQuestionAnswering.ipynb** → Build QA systems  
- **CausalLLM_All.ipynb** → Causal LLMs for text generation  
- **Classification_Sentiment_Analysis_Custom_Infer_Pipeline_HuggingFace_all_in_one.ipynb** → Sentiment analysis  
- **NERLLM_train_infer_pipeline.ipynb** → Named Entity Recognition  
- **PipeLine_NLP_All.ipynb** → Multi-task NLP pipeline  
- **Seq2SeqLM.ipynb** → Sequence-to-sequence models (translation, summarization)  

### 🔊 Audio
- **Audio_Wav2Vec2** → Automatic Speech Recognition with Hugging Face Wav2Vec2  

### 🖼️ Vision
- **ViT_Classification** → Vision Transformer for image classification  

Both Audio & Vision experiments were run locally with:
- **Trainer API** → Full training/fine-tuning loop  
- **Pipeline API** → Quick inference  



## ⚙️ LLM Environment Setup

To make setup reproducible, this repo includes a **PowerShell script**:  
📄 `LLM Environment Setup/setup-llm.ps1`  

It automates:
- Project folder creation  
- Virtual environment setup (Python 3.10)  
- Installing **Jupyter, Transformers, Datasets, Accelerate**  
- Installing **PyTorch GPU (CUDA 11.8)**  
- Registering as a Jupyter kernel  

Run with:
```powershell
.\setup-llm.ps1
````



## 🚀 Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained("your-username/myHuggingFaceLLM_Training-Inference-Pipeline-All")
model = AutoModelForCausalLM.from_pretrained("your-username/myHuggingFaceLLM_Training-Inference-Pipeline-All")

# Inference pipeline
gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = gen("Your prompt here", max_length=50)
print(output)
```

---

## 📊 Evaluation & Metrics

* Metrics used: **accuracy, perplexity, BLEU, ROUGE, etc.**
* Supports both **Trainer evaluation** and **pipeline-based quick checks**

---

## ⚠️ Bias, Risks & Ethical Considerations

* Models may inherit biases from datasets
* Potential risks: misinformation, offensive/biased outputs
* Mitigations:

  * Use domain-specific fine-tuning for sensitive tasks
  * Apply dataset filtering
  * Keep human-in-the-loop for production scenarios

---

## 🧩 Requirements

* Python **3.10**
* Hugging Face: `transformers`, `datasets`, `accelerate`, `evaluate`
* PyTorch with CUDA (for GPU acceleration)
* Jupyter Notebook / Jupyter Lab

Install all dependencies:

```bash
pip install transformers datasets accelerate evaluate jupyter torch torchvision torchaudio
```

---

## ✅ Intended Uses

* Fine-tuning Hugging Face base models on custom datasets
* Running inference for text, audio, and vision tasks
* Educational use to understand how end-to-end pipelines are built

---

## 📜 License

This repository is open-source under the **MIT License**.

---

## 👤 Author

**noobhacker02 (Talha Shaikh)**

* 🌐 Freelance Developer — Web, Data, AI/ML
* 💻 Focus: NLP, Computer Vision, LLM pipelines
* 📩 Contact: [talhatalha1971@gmail.com](mailto:talhatalha1971@gmail.com)


