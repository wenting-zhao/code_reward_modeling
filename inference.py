from typing import Dict, List
from datasets import Dataset
from vllm import LLM, SamplingParams
from utils import generate_prompt


def generate_predictions(
    model_name: str, dataset_name: str, dataset: Dataset, temperature: float = 1.0, n: int = 1
) -> List[List[str]]:
    """Generate predictions for a given dataset using a specified language model and
    sampling parameters. The function loads the dataset, constructs prompts from
    each example, and obtains generated predictions. The resulting predictions are
    then added as a new column to the dataset.

    Args:
    ----
        model_name (str): Name of the model to use for generation.
        dataset_name (str): Name of the dataset to use for generation.
        dataset (Dataset): The Dataset object.
        temperature (float, optional): Temperature setting for the model's
            sampling strategy. Default is 1.0.
        n (int, optional): Number of sampling runs per prompt. Default is 1.

    Returns:
    -------
        predictions (List[List[str]]): Predictions on the dataset.

    """
    sampling_params = SamplingParams(n=n, temperature=temperature, max_tokens=512)
    llm = LLM(model=model_name)

    prompts: List[List[Dict]] = []
    for example in dataset:
        if "mbpp" in dataset_name:
            prompt = example["text"]
            test = "\n".join(example["test_list"])
        else:
            prompt = example["prompt"]
            test = example["test"]
        prompt = generate_prompt(prompt, test)
        prompts.append([{"role": "user", "content": prompt}])

    outputs = llm.chat(prompts, sampling_params)

    results: List[List[str]] = []
    for output in outputs:
        generated_texts = [one.text for one in output.outputs]
        results.append(generated_texts)
    return results