# ECE570 Project - Reimplementation of Auto-CoT: Automatic Chain of Thought Prompting in Large Language Models (ICLR 2023)


## Introduction
This repository contains the reimplementation of the Auto-CoT (Automatic Chain of Thought) prompting method for large language models, as proposed in the ICLR 2023 paper. The goal of this project is to enhance the performance of language models on various tasks by utilizing a chain of thought prompting approach. The codebase includes scripts for generating demos and performing inference.

## File Structure
```
/dataset          # Contains benchmark datasets
/log              # Logs used for demonstration generation
/demos            # Directory to save demo outputs
/experiment       # Directory for inference outputs
run_demo.py       # Script to construct demos
run_inference.py  # Script to run inference on datasets
```

## Requirements and Setups

Python>=3.8
```
pip3 install torch torchtext
pip3 install -r requirements.txt
```

### Additional Requirements
- OpenAI API token to access GPT-4o-mini
- HuggingFace token to access Meta-Llama/Llama-3.2-3B-Instruct


## Datasets

Benchmark datasets are stored in the `/dataset` and `/log` folders.

## Instructions

Construct Demos:

```
python run_demo.py \
--task coin_flip \
--pred_file log/coin_flip_zero_shot_cot.log \
--demo_save_dir demos/coin_flip
```

Run inference:

Using gpt:
```
python run_inference.py \
--dataset coin_flip \
--demo_path demos/coin_flip \
--output_dir experiment/coin_flip \
--llm_model gpt
```

Using llama:
```
python run_inference.py \
--dataset coin_flip \
--demo_path demos/coin_flip \
--output_dir experiment/coin_flip
```

## Evaluation and Results
The `/demos` directory contains demo outputs, including clustering figures for all 9 benchmarks. The results are categorized into two sections:
- **Original implementation** (files prefixed with `benchmarkname_original`)
- **Reimplementation** (files prefixed with `benchmarkname_new`)

The `/experiment` directory contains the LLM output, with 4 output files for each of the 9 benchmarks:
- `benchmarkname_llama_new`
- `benchmarkname_llama_original`
- `benchmarkname_gpt_new`
- `benchmarkname_gpt_original`