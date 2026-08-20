## FinanceReasoning

This project was developed with the assistance of modern AI-powered development tools, including Cursor IDE and Tongyi Qianwen. All code has been carefully reviewed to ensure originality and compliance with best practices. The implementation represents original work by the authors.

The data and code for the paper `FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging`.

**FinanceReasoning** is a a novel benchmark designed to evaluate the reasoning capabilities of large reasoning models (LRMs) in financial numerical reasoning problems. 

## FinanceReasoning Dataset
Based on the difficulty of reasoning, we divided the problems into three subsets: *Easy* (1,000 examples), *Medium* (1,000 examples), and *Hard* (238 examples). 

The dataset is provided in json format and contains the following attributes at the `data` directory:

```json
{
    "question_id": "[string] Unique identifier for the question",
    "question": "[string] The question text, typically a financial data analysis problem",
    "context": "[string] Background information for the question, including tabular data in Markdown format",
    "statistics": {
        "number_statistics": "[object] Statistics about numbers, including count of numbers in the question",
        "operator_statistics": "[object] Statistics about operator usage, tracking frequency of different operators",
        "code_statistics": "[object] Code-related statistics, such as number of code lines"
    },
    "python_solution": "[string] Python solution code written by financial experts, with clear variable names and execution logic",
    "ground_truth": "[number / boolean] The standard answer, typically the result of executing the Python solution",
    "difficulty": "[float] Difficulty coefficient of the question, higher values indicate greater difficulty",
    "level": "[string] Difficulty level classification of the question (e.g., hard, medium, easy)",
    "source": "[string] Source identifier of the question"
}
```

## Financial Functions Library

The financial functions library is a collection of financial functions that are used to solve the financial numerical reasoning problems. It is provided in json format and contains the following attributes at the `data/functions` directory:

```json
{
    "function_id": "[string] Unique identifier for the function",
    "function": "[string] The function code",
    "function_docstring": "[string] The docstring of the function"
}
```

## Financial Documents Library

The financial documents library is a collection of financial documents that are used to solve the financial numerical reasoning problems. It is provided in json format and contains the following attributes at the `data/documents` directory:

```json
{
    "document_id": "[string] Unique identifier for the document",
    "document": "[string] The document text",
    "document_docstring": "[string] The docstring of the document"
}
```

## Experiments (Main Results)

### Environment Setup
You can install the dependencies by the following command:
```bash
pip install -r requirements.txt
```

### Configuration
The `config/config.yaml` file controls all aspects of inference and evaluation:
- Inference settings (e.g., dataset, subset, model, prompt type)
- Evaluation settings
- Model configurations (API keys, base URLs, sampling parameters)

### Running Inference
We support inference with various LLM models through two approaches:

1. **Configuration-based Inference**
   ```bash
   python inference.py --config config/config.yaml
   ```
   This method uses the configuration file to specify model settings, dataset parameters, and inference options.

2. **Batch API Inference**
   ```bash
   python utils/openai_batch.py \
     --dataset "FinanceReasoning" \
     --subset "hard" \
     --prompt "cot" \
     --model "your_model_id" \
     --api_key "your_api_key" \
     --base_url "your_base_url"
   ```
   This method allows you to get 50% discount on the openai inference cost.

### Model Output
Inference results are stored in the `results` directory, organized by:
- Dataset name
- Dataset subset (`hard`, `medium`, `easy`)
- Prompt type
- Model name

### Automated Evaluation
Evaluate model performance using:
```bash
python evaluation.py --config config/config.yaml
```

## Experiments (Additional Results)
### Complete the .env file

Replace the `API_KEY` and `BASE_URL` in the `.env` file with your own API key and base URL.

### Run Function Retrieval Server

```bash
python serve_retriever_function.py
```

### Run Section Retrieval Server (Optional for Passage Retrieval, replace the port if necessary)

```bash
python serve_retriever_section.py
```

### Run RAG and Model Combination Parallel Inference

```bash
python rag_parallel.py
```

You can set the arguments as follows:

dataset = 'FinanceReasoning'
subset = 'hard'
prompt_type = 'cot_rag'
model_name = 'gpt-4o-2024-11-20'
model_name_file = 'gpt-4o-2024-11-20'
llm_instruct = True
use_article = True
use_reasoning = False
use_retrieved_cache = False
retrieved_type = 'function'
judge_useful_functions = True
use_useful_cache = True
top_k = 30
input_file = f'./data/{dataset}/{subset}.json'

### Results of RAG and Model Combination Parallel Inference

The CoT outputs are stored in the `results/FinanceReasoning/hard/raw_cot_outputs` and `results/FinanceReasoning/hard/processed_cot_outputs` directory.
The CoT results are stored in the `results/FinanceReasoning/hard/results/hard_cot_results.json`

The PoT outputs are stored in the `results/FinanceReasoning/hard/raw_pot_outputs` and `results/FinanceReasoning/hard/processed_pot_outputs` directory.
The PoT results are stored in the `results/FinanceReasoning/hard/results/hard_pot_results.json`