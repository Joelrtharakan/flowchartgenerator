# Flowchart Generation System using Fine-Tuned Phi-3

## Overview

This project implements an end-to-end system for generating Mermaid flowcharts from natural language prompts using a fine-tuned large language model.

The system is built around the **Phi-3 Mini Instruct** model and enhanced using **QLoRA-based fine-tuning** on domain-specific data.

### Features

* Natural language to Mermaid flowchart generation
* Continuous dataset expansion via PostgreSQL
* Automated retraining pipeline *(planned via monitor system)*
* Deployment on Azure Virtual Machine using FastAPI
* Real-time logging and monitoring

---

## Architecture

The system consists of three major components:

### 1. Training Pipeline

* Fetches prompt–flowchart pairs from PostgreSQL
* Applies filtering and formatting
* Fine-tunes Phi-3 using QLoRA (4-bit quantization)
* Saves LoRA adapters (lightweight model updates)

### 2. Inference API

* FastAPI-based backend deployed on Azure VM
* Loads base Phi-3 model + LoRA adapter
* Accepts prompts and returns Mermaid flowcharts
* Provides logging endpoints for monitoring

### 3. Deployment Environment

* Azure Virtual Machine (CPU-based inference)
* Python virtual environment
* Model artifacts stored locally
* API exposed via Uvicorn

---

## Project Structure

```
.
├── app.py                  # FastAPI inference server
├── train_phi3.py           # Fine-tuning script (GPU required)
├── phi3-finetuned/         # LoRA adapter files
├── model.zip               # Zipped adapter (for transfer)
├── requirements.txt        # Python dependencies
├── api.log                 # API logs
├── monitor.log             # Training/monitor logs (future use)
├── current_model.txt       # Points to active model path
└── venv/                   # Virtual environment
```

---

## Model Details

* **Base Model:** microsoft/phi-3-mini-4k-instruct
* **Fine-tuning Method:** QLoRA (4-bit quantization)

### Frameworks Used

* Transformers

* PEFT (LoRA)

* Accelerate

* **Training Device:** GPU required (local, Colab, or cloud GPU)

* **Inference Device:** CPU (Azure VM)

---

## Training Pipeline

### Requirements

* CUDA-enabled GPU
* PostgreSQL database with training data

### Environment Variables

```
DB_HOST
DB_USER
DB_PASSWORD
DB_NAME (optional)
```

### Run Training

```bash
python train_phi3.py
```

### Output

LoRA adapter will be saved in:

```
./phi3-finetuned/
```

---

## Deployment on Azure VM

### 1. Setup VM

* Ubuntu 22.04 or later
* Open port **8000** in Azure Network Security Group

---

### 2. Transfer Files

```bash
scp app.py azureuser@<VM_IP>:~
scp -r phi3-finetuned azureuser@<VM_IP>:~
```

---

### 3. Setup Environment

```bash
sudo apt update
sudo apt install python3.12-venv unzip -y

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install fastapi uvicorn torch transformers peft accelerate psycopg2-binary
```

---

### 4. Run API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

### 5. Access API

* **Swagger UI:**

  ```
  http://<VM_IP>:8000/docs
  ```

* **Logs:**

  ```
  http://<VM_IP>:8000/logs/api
  http://<VM_IP>:8000/logs/monitor
  ```

---

## API Usage

### Endpoint

```
POST /generate
```

### Request Body

```json
{
  "prompt": "Deploy model in Azure"
}
```

### Response

```json
{
  "flowchart": "flowchart TD\nA --> B ..."
}
```

---

## Logging System

The system logs:

* Incoming prompts
* Generated outputs (truncated)
* Runtime events

### Log Files

```
api.log
monitor.log
```

Logs can also be accessed via API endpoints.

---

## Model Versioning

The system uses:

```
current_model.txt
```

This file determines which LoRA adapter is currently active.

### Benefits

* Switch models without modifying code
* Support multiple trained versions
* Enable rollback capability

---

## Future Enhancements

* Automatic retraining using `monitor.py`
* Dataset growth detection
* Model version management
* Frontend dashboard for logs and usage
* CI/CD pipeline for deployment
* GPU-based Azure deployment option

---

## Key Design Decisions

* QLoRA used to reduce GPU memory usage during training
* LoRA adapters used instead of full model retraining
* CPU inference chosen for cost efficiency
* FastAPI used for lightweight deployment
* Logging exposed via API for observability

---

## Limitations

* Training requires GPU (cannot run on CPU)
* CPU inference is slower compared to GPU
* No automated retraining pipeline yet (manual trigger required)
* No UI dashboard implemented yet

---

## Conclusion

This project demonstrates a scalable pipeline for:

* Fine-tuning LLMs on structured generation tasks
* Deploying them efficiently on cloud infrastructure
* Maintaining extensibility for continuous learning systems

It is designed to evolve into a fully automated ML system with monitoring, retraining, and version control.
