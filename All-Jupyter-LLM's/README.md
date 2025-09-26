# 🎯 Local Model Experiments — Audio & Vision

This project demonstrates training and inference of two different model types using the **Hugging Face ecosystem**:  

1. **Audio Model (Wav2Vec2)** → for speech/audio tasks  
2. **Vision Model (ViT - Vision Transformer)** → for image classification  

Both models were tested locally on my system with **Trainer API** and **Pipeline API**, ensuring flexible workflows for experimentation.

---

## 📂 Models Covered

### 🔊 1. Wav2Vec2 (Audio)
- **Task:** Automatic Speech Recognition (ASR) / Audio feature extraction  
- **Model:** `facebook/wav2vec2-base` (or custom fine-tuned variant)  
- **Workflows:**
  - **Trainer**: Fine-tuning and evaluation on audio datasets.  
  - **Pipeline**: Quick inference for speech-to-text.

### 🖼️ 2. ViT (Vision Transformer)
- **Task:** Image classification  
- **Model:** `google/vit-base-patch16-224` (or custom fine-tuned variant)  
- **Workflows:**
  - **Trainer**: Training and evaluating on image datasets.  
  - **Pipeline**: Direct classification of input images.

---

## ⚙️ Pipelines vs Trainer

| Approach     | Description | When to Use |
|--------------|-------------|-------------|
| **Trainer**  | Full training loop (fine-tuning, evaluation, metrics) | When customizing or retraining models on new datasets |
| **Pipeline** | Prebuilt inference wrapper for quick predictions | When you need fast inference without custom training |

---

## 🖥️ Example Usage

### Wav2Vec2 Pipeline
```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base")
result = asr("path_to_audio.wav")
print(result["text"])
ViT Pipeline
python
Copy code
from transformers import pipeline
from PIL import Image

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
image = Image.open("sample.jpg")
print(classifier(image))
Trainer API (General Pattern)
python
Copy code
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
🧩 Requirements
Python 3.10+

Hugging Face Transformers, Datasets, Accelerate

PyTorch with CUDA (for GPU acceleration)

Jupyter (if running notebooks)

Install dependencies:

bash
Copy code
pip install transformers datasets accelerate evaluate jupyter torch torchvision torchaudio
✅ Key Learnings
Wav2Vec2 is excellent for audio tasks like speech-to-text.

ViT provides strong performance for image classification tasks.

Hugging Face makes it easy to switch between Trainer (custom training) and Pipeline (quick inference) for different needs.

📜 License
Open-source for learning and experimentation.

yaml
Copy code

