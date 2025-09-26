# 🛠️ Simple LLM Environment Setup (PowerShell Script)

This PowerShell script automates the setup of a **Python environment for Large Language Model (LLM) projects**.  
It creates a project folder, initializes a virtual environment, installs the required libraries (Hugging Face, PyTorch, Jupyter), and registers the environment as a Jupyter kernel.  

---

## 🚀 Features
- Creates a new **project directory** inside your `Downloads\LLm\All-Jupyter-LLM's` folder.
- Sets up a **Python 3.10 virtual environment**.
- Installs:
  - **Jupyter Notebook / Jupyter Lab**  
  - Hugging Face ecosystem: `transformers`, `datasets`, `evaluate`, `accelerate`
  - **PyTorch GPU (CUDA 11.8 build)** stack (`torch`, `torchvision`, `torchaudio`)
- Registers the environment as a **Jupyter kernel** (so you can select it directly in notebooks).

---

## 📂 Project Workflow
When you run the script:
1. You’ll be asked for a **new project folder name**.  
   Example: `MyExperiment` → will be created at  
   `C:\Users\Talha Shaikh\Downloads\LLm\All-Jupyter-LLM's\MyExperiment`

2. You’ll be asked for a **virtual environment name**.  
   Example: `llm_env`

3. The script will then:
   - Create and activate the virtual environment.
   - Install all required packages.
   - Register the kernel with Jupyter.

---

## 🖥️ Usage

### 1. Save the Script
Save your script as `setup-llm.ps1`.

### 2. Run in PowerShell
Open **PowerShell** and navigate to the script location. Then run:

```powershell
.\setup-llm.ps1
