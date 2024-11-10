# ECE570 Project - Reimplementation of Auto-CoT: Automatic Chain of Thought Prompting in Large Language Models (ICLR 2023)

## Requirements

Python>=3.8
```
pip3 install torch torchtext
pip3 install -r requirements.txt
```

## Datasets

Download the datasets from the following:

```
https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/dataset
https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/log
```

## Instructions

Construct Demos:

```
python run_demo.py \
--task multiarith \
--pred_file log/multiarith_zero_shot_cot.log \
--demo_save_dir demos/multiarith
```

Run inference:

```
python run_inference.py \
--dataset multiarith \
--demo_path demos/multiarith \
--output_dir experiment/multiarith
```