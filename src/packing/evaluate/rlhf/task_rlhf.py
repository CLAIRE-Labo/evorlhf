from typing import List, Dict, Any, Tuple, Callable
import numpy as np
import sys
import logging
import random
import copy
import traceback
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
        return "Dataset name"
    elif set == "testset":
        return "Testset name"
    else:
        raise ValueError(f"Unknown set: {set}")


def get_model_scores(model, generations):
    """Return a list of scores given a model and a list of generations."""
    return [model(generation) for generation in generations]

def evaluate_kendall_correlation(true_rm, heuristic, train_data, sample_size=100, seed=42):
    sampled_generations = random.sample(train_data, sample_size)
    true_rm_scores = get_model_scores(true_rm, sampled_generations)
    heuristic_scores = get_model_scores(heuristic, sampled_generations)
    tau, p_value = kendalltau(true_rm_scores, heuristic_scores)
    return tau, p_value

def evaluate_func(cfg, dataset_name, function_class) -> FunctionClass:

    def _get_instances(dataset_name):
        pass

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
        function_class.score = cfg.failed_score
        function_class.true_score = cfg.failed_score
        function_class.fail.exception = tb_str
        return function_class
    
    instances = _get_instances(dataset_name)
    rewards_heuristic = []
    rewards_rm = []
    for instance in instances:
        generation = instance['generation']
        reward = func_from_llm(generation)
        rewards_heuristic.append(reward)
        reward_rm = instance['reward']
        rewards_rm.append(reward_rm)
    # Calculate the correlation
    tau, p_value = kendalltau(rewards_rm, rewards_heuristic)
    # Calculate the score
    score = tau
    # Set the score
    function_class.score = score
    function_class.true_score = score

    return function_class


append_prompt = ""
system_prompt = ""

TASK_REGISTRY.register(
    "rlhf",
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