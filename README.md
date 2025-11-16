# COMP4222_Group6
# Evaluation Scripts

This repository contains scripts for evaluating models with and without RAG (Retrieval-Augmented Generation). The following scripts are provided:

## Main Files

1. **`No_Rag_main.py`**
   - This script is used for evaluation **without RAG**.

2. **`Hipporag_main.py`**
   - This script is used for evaluation **with HippoRAG**.

## Instructions

1. Navigate to the directory containing the scripts.

2. Set the environment variables:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   export OPENAI_API_KEY=<your_openai_api_key>  # Replace <your_openai_api_key> with your DeepSeek API key

3. Running the scripts:
   
   - For No_Rag_main.py:
   Open the file and manually update the dataset name inside the script to the desired dataset.
   Run the script:
   ```bash
   python No_Rag_main.py
   ```
   
   - For Hipporag_main.py:
   Run the script directly by specifying the dataset name as a command-line argument:
   ```bash
   python Hipporag_main.py --<dataset_name>
   ```
4. Running experiments:

   - To run experiments with custom configurations, modify the configs in /src/hipporag/utils
   
## Note: 
All available datasets for evaluation are stored in the /reproduce/dataset directory. You only need to include the dataset name in the script or command.
