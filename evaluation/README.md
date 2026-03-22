# CXReasonBench Evaluation

This README provides instructions for running the CXReasonBench evaluation pipeline, including environment setup, dataset download, and execution of the evaluation scripts.

For the details of the evaluation pipeline, refer to Appendix B: Details of CXReasonBench in the [paper](https://arxiv.org/pdf/2505.18087).

## Environment Setup


### Conda Environment

Follow the steps below to set up the required environment for running benchmark evaluation:

```bash
# Create a conda environment
conda create -n cxreasonbench python=3.9

# Activate the environment
conda activate cxreasonbench

# Install required packages
pip install -r requirements.txt
```
---
### API Setup

Both MIMIC-CXR-JPG and CXReasonBench require credentialed access via PhysioNet.

For responsible use of online services and LLMs on these datasets, please refer to the [PhysioNet Responsible Use Guidelines](https://physionet.org/news/post/llm-responsible-use).

To comply with these guidelines, the evaluation scripts must be executed on authorized servers, depending on the model type:

- **Gemini** models run on **Vertex AI**
  - Required setup: see [Vertex AI Generative AI Quickstart](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart?usertype=adc)
- **GPT** models run on **Azure OpenAI**
    - Required setup: see [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/create-resource?pivots=web-portal) 

The benchmark scripts connect to these servers using the following command-line arguments in `evaluate_reasoning.py`:

#### Vertex AI (Gemini)
```
--GOOGLE_CLOUD_PROJECT <your-project-id> \
--GOOGLE_CLOUD_LOCATION <your-project-location> \
--GOOGLE_GENAI_USE_VERTEXAI True
```
#### Azure OpenAI (GPT)
```
--gpt_endpoint <your-endpoint> \
--gpt_api_key <your-api-key> \
```

## Dataset Download

The benchmark requires two datasets, both of which are available on [PhysioNet](https://physionet.org/) and require credentialed access.

### 1. Get PhysioNet Access
To download the datasets, you must first obtain credentialed access from PhysioNet.  
Follow the instructions here to complete the credentialing process:  
👉 [https://physionet.org/settings/credentialing/](https://physionet.org/settings/credentialing/)

Once your account is approved, log in to PhysioNet and download the following datasets.


### 2. Required Datasets

| Dataset | Description | Download Link |
|----------|--------------|----------------|
| **MIMIC-CXR-JPG** | Chest X-ray images in JPEG format derived from MIMIC-CXR DICOM data | [https://physionet.org/content/mimic-cxr-jpg/](https://physionet.org/content/mimic-cxr-jpg/) |
| **CXReasonBench** | A benchmark for evaluating the reasoning process of models in chest X-ray diagnosis. | [https://physionet.org/content/chexstruct-cxreasonbench/](https://physionet.org/content/chexstruct-cxreasonbench/) |


### 3. Directory Structure After Download

After downloading, your local directory structure should look like this:

```text
<path_to_physionet_download_dir>/
├─ physionet.org/
│  ├─ files/
│  │  ├─ mimic-cxr-jpg/
│  │  │  └─ <version>/files/
│  │  └─ chexstruct-cxreasonbench/
│  │     └─ <version>/CXReasonBench/
```

Inside the `CXReasonBench` folder, the dataset is provided as a .zip file. Please unzip it before running the evaluation scripts.

## Run the Evaluation Pipeline


The `main.py` script runs the **full evaluation pipeline**, which includes:
1. Reasoning evaluation (`evaluate_reasoning.py`)  
2. Guidance evaluation (`evaluate_guidance.py`)  
3. Metric calculation (`metric.py`)

All steps are executed sequentially using `main.py` in the `Benchmark/evaluation` folder.

---
### Configurations
`main.py` uses pre-defined arguments for the evaluation scripts. 

Below are the key configuration options:

| Argument | Description | Example |
|----------|-------------|---------|
| `--model_id` | Identifier of the model to evaluate | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `--model_path` | Path to pretrained model weights (downloaded from HuggingFace) | `<path_to_downloaded_model>/snapshots/<hash>` |
| `--tensor_parallel_size` | Number of tensor-parallel GPUs (used for vLLMs multi-GPU setup) | `2` |
| `--model_id4scoring` | Judge model used to evaluate model responses against the reference answers (default) | `gemini-2.0-flash` |
| `--cxreasonbench_base_dir` | Path to the CXReasonBench dataset | `<path_to_physionet_download_dir>/physionet.org/files/chexstruct-cxreasonbench/<version>/CXReasonBench` |
| `--mimic_cxr_base` | Path to the MIMIC-CXR-JPG dataset | `<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/files` |
| `--save_base_dir` | Base directory for evaluation results; folder where outputs will be saved | `result` |
---
### Run the Pipeline

Once all arguments are properly configured, execute the full pipeline using:
```
# Activate the environment
conda activate cxreasonbench

# Navigate to Evaluation Folder
cd CXReasonBench/Benchmark/evaluation

# Run the script
CUDA_VISIBLE_DEVICES=<comma-separated GPU IDs> python main.py
```
---
### Output Structure
After running `main.py`, all outputs will be stored under the directory specified by `--save_base_dir`:
```text
<save_base_dir>/
├─ inference/
│  ├─ reasoning/<model_id>/       # Model responses from evaluate_reasoning.py
│  └─ guidance/<model_id>/        # Model responses from evaluate_guidance.py
└─ scoring/
   ├─ reasoning/<model_id4scoring>/<model_id>/   # Evaluation results judged by <model_id4scoring> (from evaluate_reasoning.py)
   └─ guidance/<model_id4scoring>/<model_id>/    # Evaluation results judged by <model_id4scoring> (from evaluate_guidance.py)
```
- `<model_id>`: The model being evaluated (e.g., Qwen/Qwen2.5-VL-7B-Instruct)
- `<model_id4scoring>`: The judge model used to evaluate model responses against the reference answers (e.g., gemini-2.0-flash)