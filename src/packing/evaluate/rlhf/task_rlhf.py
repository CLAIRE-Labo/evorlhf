from typing import List, Dict, Any, Tuple, Callable
import numpy as np
import sys
import logging
import random
import copy
import traceback
from datasets import load_dataset, load_from_disk
from packing.logging.function_class import FunctionClass
from scipy.stats import kendalltau
from packing.evaluate.registry import TASK_REGISTRY

GENERAL_IMPORTS = '''
import random
import numpy
import numpy as np
from itertools import product
import math
import scipy
import scipy.stats
import scipy.special
import copy
'''

# TODO: use the cfg instead of hardcoding the data path
DATA_PATH = '/iopsstor/scratch/cscs/nevali/projects/evorlhf/data/Magpie-Air-DPO-100K-v0.1'

def get_initial_func(cfg) -> Tuple[Callable, str]:
    # Function to be evolved.
    def heuristic_rm(generation: str) -> int:
        """Returns score for a given generation.
        Args:
            generation (str): Generation to evaluate.
        
        Returns:
            int: Score for the generation.
        """
        return len(generation)

    initial_function = heuristic_rm
    function_str_to_extract = "heuristic_rm"

    return initial_function, function_str_to_extract


# Function to create the input.
def generate_input(cfg, set: str) -> str:
    if set == "train":
        return "Magpie-Align/Magpie-Air-DPO-100K-v0.1"
    elif set == "testset":
        return "Magpie-Align/Magpie-Air-DPO-100K-v0.1"
    else:
        raise ValueError(f"Unknown set: {set}")


def evaluate_func(cfg, dataset_name, function_class) -> FunctionClass:

    def _get_instances(dataset_name):
        # Load the dataset from Hugging Face
        # dataset = load_dataset(dataset_name, split='train')
        # Load the dataset from scratch folder (I downloaded only the test data):
        dataset = load_from_disk(DATA_PATH)
        # Consider only 100 random samples for evaluation
        dataset = dataset.shuffle(seed=cfg.seed).select(range(100))
        # Check if the dataset has the required columns
        if 'responses' not in dataset.column_names or 'rewards_armorm' not in dataset.column_names:
            raise logging.error(f"Dataset {dataset_name} does not contain the required columns.")
        # Check if the dataset is empty
        if len(dataset) == 0:
            raise logging.error(f"Dataset {dataset_name} is empty.")
        # Convert the dataset to a list of dictionaries
        instances = []
        for i in range(len(dataset)):
            instance = {
                'responses': dataset[i]['responses'],
                'rewards': dataset[i]['rewards_armorm'],
            }
            instances.append(instance)
        return instances

    priority_func_str = function_class.function_str
    imports = function_class.imports_str

    # Execute imports and the function.
    try:
        # Create a shared gloabls dictionary
        globals_dict = {}
        # Execute general imports
        exec(GENERAL_IMPORTS, globals_dict)
        # Execute the imports into the globals dictionary
        exec(imports, globals_dict)
        # Execute the perturbation function string using the same globals dictionary
        load_dict = {}
        exec(priority_func_str, globals_dict, load_dict)
        # Extract the function from the dictionary
        func_from_llm = load_dict.get(cfg.function_str_to_extract)
        assert func_from_llm is not None, f"Function {cfg.function_str_to_extract} not found in the loaded dictionary"
    except Exception as e:
        tb_str = traceback.format_exc()
        function_class.fail_flag = 1
        function_class.fail.reason_imports = 1
        function_class.score = cfg.task.failed_score
        function_class.true_score = cfg.task.failed_score
        function_class.fail.exception = tb_str
        return function_class
    
    logging.info(f"Function {cfg.function_str_to_extract} imported successfully.")
    instances = _get_instances(dataset_name)
    rewards_heuristic = []
    rewards_rm = []
    for instance in instances:
        generations = instance['responses']
        rewards = instance['rewards']
        for i in range(len(generations)):
            generation = generations[i]
            reward_rm = rewards[i]
            try:
                reward_heuristic = func_from_llm(generation)
                rewards_heuristic.append(reward_heuristic)
            except Exception as e:
                logging.info(f"Function {cfg.function_str_to_extract} failed to execute: {e}")
                tb_str = traceback.format_exc()
                function_class.fail_flag = 1
                function_class.score = cfg.task.failed_score
                function_class.true_score = cfg.task.failed_score
                function_class.fail.exception = tb_str
                return function_class
            rewards_rm.append(reward_rm['score'])
    # Calculate the correlation
    try:
        tau, p_value = kendalltau(rewards_rm, rewards_heuristic)
        # Check if Kendall's tau calculation resulted in NaN
        if tau is None or p_value is None or np.isnan(tau) or np.isnan(p_value):
            logging.warning(f"[task_rlhf.py] Kendall's tau calculation resulted in NaN. tau={tau}, p_value={p_value}")
            logging.warning(f"[task_rlhf.py] Rewards RM (len {len(rewards_rm)}): {rewards_rm}")
            logging.warning(f"[task_rlhf.py] Rewards Heuristic (len {len(rewards_heuristic)}): {rewards_heuristic}")
            function_class.fail_flag = 1
            function_class.score = cfg.task.failed_score
            function_class.true_score = cfg.task.failed_score
            function_class.fail.exception = "Kendall's tau resulted in NaN"
            return function_class
        function_class.score = tau
        function_class.true_score = tau
        function_class.p_value = p_value
        function_class.fail_flag = 0
        function_class.correct_flag = 1

    except Exception as e:
        logging.info(f"Heuristic is not able to compute Kendall's tau: {rewards_heuristic}, {len(rewards_heuristic)}")
        tb_str = traceback.format_exc()
        logging.info(f"[task_rlhf.py] Function {cfg.function_str_to_extract} failed to execute: {e}")
        function_class.fail_flag = 1
        function_class.score = cfg.task.failed_score
        function_class.true_score = cfg.task.failed_score
        function_class.fail.exception = tb_str
    
    return function_class


append_prompt = """
You are tasked with discovering a new function, heuristic_reward(), that closely approximates the behavior of the learned reward model while being significantly simpler and more interpretable.

To achieve this, follow these guidelines:

1. Understand the True Reward Model
   - The true reward model is a complex function learned through reinforcement learning from human feedback (RLHF).  
   - Your goal is to uncover a simpler heuristic that captures its core principles without unnecessary complexity.

2. Optimize for Fidelity
   - Your heuristic should produce rewards that are as close as possible to the true reward model.  
   - The performance of your heuristic will be evaluated based on how well it minimizes the expected difference from the true reward model across a variety of inputs.

3. Prioritize Simplicity and Interpretability
   - Avoid overfitting to small variations—aim for a general rule that explains most of the reward function’s behavior.  
   - Prefer human-readable mathematical expressions with fewer terms and operations.  

4. Think Creatively
   - Do not simply copy or rephrase parts of the learned reward model.  
   - Look for underlying patterns or principles that define the reward function and express them in a simpler form.

5. Experiment and Improve
   - Analyze past attempts: what features contributed most to performance?  
   - Try new variations and optimizations that improve accuracy while keeping the heuristic simple.  
   - Your goal is to iteratively refine heuristic_reward() until it consistently achieves high alignment with the true reward model.

To summarize, your task is to write a new function named heuristic_reward() that approximates the learned reward model with minimal complexity while maintaining high accuracy. Your success will be measured by how well it generalizes across different inputs while remaining interpretable.
"""
system_prompt = "You are helpful, excellent and innovative problem solver specializing in mathematical optimization and algorithm design. You are an expert in writing Python functions."


TASK_REGISTRY.register(
    "evorlhf",
    generate_input=generate_input,
    evaluate_func=evaluate_func,
    get_initial_func=get_initial_func,
    system_prompt=append_prompt,
    append_prompt=system_prompt,
)


if __name__ == '__main__':
    
    # Why here?
    def priority():
        pass