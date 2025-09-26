param()

Write-Host "=== Simple LLM Environment Setup ===`n"

# Ask for new project folder name
$dirName = Read-Host "Enter new project folder name (will be created inside Downloads\LLm\All-Jupyter-LLM's)"
$basePath = "C:\Users\Talha Shaikh\Downloads\LLm\All-Jupyter-LLM's"
$projectPath = Join-Path $basePath $dirName

# Create folder if it doesn't exist
if (-not (Test-Path $projectPath)) {
    New-Item -ItemType Directory -Path $projectPath | Out-Null
}
Set-Location $projectPath
Write-Host "Project directory set to: $projectPath`n"

# Ask for environment name
$envName = Read-Host "Enter new environment name"
if ([string]::IsNullOrWhiteSpace($envName)) {
    $envName = "test_env"
}

# 1. Create virtual environment
Write-Host "`n[1] Creating virtual environment $envName ..."
py -3.10 -m venv $envName

# 2. Activate environment
Write-Host "[2] Activating environment ..."
& .\${envName}\Scripts\Activate.ps1

# 3. Upgrade pip
Write-Host "[3] Upgrading pip ..."
python -m pip install --upgrade pip

# 4. Install Jupyter + HuggingFace libs
Write-Host "[4] Installing Jupyter + HuggingFace libraries ..."
pip install jupyter ipykernel transformers datasets evaluate accelerate

# 5. Install GPU PyTorch stack
Write-Host "[5] Installing GPU-enabled PyTorch stack ..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 6. Register Jupyter kernel
Write-Host "[6] Registering environment as Jupyter kernel ..."
python -m ipykernel install --user --name $envName --display-name $envName

Write-Host "`n[SUCCESS] Environment '$envName' is ready and registered!"
