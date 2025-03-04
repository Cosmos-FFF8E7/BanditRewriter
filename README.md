# BanditRewriter
The code repository for the paper *BanditRewriter: Training-free Model Adaptive Prompt Optimization for T2I Generation*
* Note: This is preliminary version of our code. The complete code to run all experiments in the paper will be added shortly.

![BR](https://github.com/Cosmos-FFF8E7/BanditRewriter/blob/main/BR.png)

## Abstract

<details><summary>CLICK for the full abstract</summary>

> In the Text-to-Image (T2I) generation task, precisely constructed prompts are crucial for fully leveraging the generative model’s capabilities and obtaining preferable image outputs. However, well-performed prompts are usually model-specific, requiring customization for each T2I model. Existing prompt optimization methods typically rely on inefficient searches, heuristic algorithms, or model-specific training, which are computationally expensive and inflexible. To address these challenges, BanditRewriter is proposed, which leverages large language models to adaptively optimize prompts for different T2I models. By incorporating upper confidence bounds and beam search, BanditRewriter achieves an effective exploration-exploitation trade-off to efficiently develop model-preferred prompt rewriting strategies without model training. Experiments on Stable Diffusion 2.1 and DALL·E 3 demonstrate that BanditRewriter can enhance text-image alignment with an 8.76% improvement in Image Reward while boosting generated image quality through a 3.56% increase in Aesthetics score.
</details>


## Setup Environment

```bash
conda create -n BanditRewriter python=3.10
conda activate BanditRewriter
pip install -r requirements.txt
```

## Key Dependencies
```bash
torch>=1.7.0
transformers>=4.25.1
diffusers>=0.16.0
sentence-transformers
clip
scikit-learn
```

## Configuration

Key hyperparameters in config.py:

### API Keys

* `--openai_key`: OpenAI API key for GPT models
* `--dalle_key`: DALL·E API key  
* `--mj_key`: Midjourney API key

### Model Configuration

* `--llm`: Choose LLM type ["gpt4", "gpt3"]
* `--t2i_model`: Choose T2I model ["sd21", "dalle3", "midjourney"]
* `--api_endpoint`: Custom API endpoint URL (optional)

### Optimization Parameters

* `--alpha`: UCB exploration parameter
* `--num_iterations`: Number of optimization iterations
* `--ucb_iterations`: UCB iterations per step  
* `--beam_width`: Beam search width
* `--batch_size`: Batch size for processing

## Usage

### Prompt Optimization Pipeline

#### Train

```bash
python main.py --mode train \
    --dataset path/to/dataset \
    --llm gpt4 \
    --t2i_model sd21 \
    --num_clusters 50 \
    --alpha 0.65 \
    --num_iterations 5 \
    --ucb_iterations 5 \
    --beam_width 5 \
    --output optimal_rewrite_prompt.txt \
    --openai_key YOUR_OPENAI_API_KEY \
    --dalle_key YOUR_DALLE_API_KEY \
    --mj_key YOUR_MIDJOURNEY_API_KEY
```

#### Inference

```bash
python main.py --mode infer \
    --t2i_model dalle3 \
    --rewrite_prompt optimal_rewrite_prompt.txt \
    --prompt "a female fluffy fox animal" \
    --openai_key YOUR_OPENAI_API_KEY \
    --dalle_key YOUR_DALLE_API_KEY
```

### Evaluation

```bash
python evaluate/run_eval.py \
    --model {sd21, dalle3, midjourney} \
    --test_data path/to/test/data \
    --metrics "clip,hps,ir,aes" \
    --output_dir eval_results \
    --openai_key YOUR_OPENAI_API_KEY \
    --dalle_key YOUR_DALLE_API_KEY \
    --mj_key YOUR_MIDJOURNEY_API_KEY 
```

## Directory Structure

```
.
├── config/
│   └── config.py          # Configuration settings
├── data/
│   └── br_dataset.py      # Data loading utilities  
├── optimize/
│   ├── bandit_optimize.py # UCB optimization
│   └── beam_search.py     # Beam search implementation
├── evaluate/
│   ├── metrics.py         # Evaluation metrics
│   └── run_eval.py        # Evaluation script
├── models/
│   ├── llm_wrapper.py     # LLM API wrapper
│   └── t2i_wrapper.py     # T2I model wrapper
├── main.py                # Main training/inference script
├── requirements.txt       # Dependencies
└── README.md
```

