"""
Inference with multiple datasets using vLLM for a single model
"""

# from transformers import Qwen2model
from transformers import Qwen2Model

import sys
sys.path.append('./')
import pandas as pd
import json
import torch
import os
import fire
from vllm import LLM, SamplingParams
import gc
from tqdm import tqdm
import time
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

OFFICIAL_JB_DIR = os.path.join(os.path.dirname(__file__), "data", "official_jailbreak", "converted")
OFFICIAL_NO_INSTRUCTION_DATASETS = {"gcg_official", "pair_official"}


def save_results(prompts, responses, output_path):
    """Save results to JSON file"""
    results = [
        {"prompt": prompt, "response": response}
        for prompt, response in zip(prompts, responses)
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")

def batch_generate_responses(prompts, llm, batch_size=128, **generation_kwargs):
    """Generate responses for a batch of prompts using vLLM"""
    sampling_params = SamplingParams(
        temperature=generation_kwargs.get('temperature', 0),
        max_tokens=generation_kwargs.get('max_new_tokens', 256),
        top_p=generation_kwargs.get('top_p', 0.9),
        top_k=generation_kwargs.get('top_k', 50),
        stop= ["<|im_end|>"]
    )
    
    all_responses = []
    
    try:
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[i:i + batch_size]


            outputs = llm.generate(batch, sampling_params)
            
            for output in outputs:
                response = output.outputs[0].text
                all_responses.append(response)
                
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return all_responses
    
    return all_responses

def main(
    model_id: str,
    device: int=0,
    max_new_tokens: int=2048,
    seed: int=42,
    do_sample: bool=True,
    top_p: float=0.9,
    temperature: float=0.6,
    top_k: int=50,
    prefix: str = "",
    Think: bool = False,
    memory_utilization: float=0.8,
    **kwargs
):
    # Set seeds
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    print(f"\nProcessing model: {model_id}")

    llm = LLM(
        model=model_id,
        #tokenizer="princeton-nlp/Llama-3-Base-8B-SFT",
        tensor_parallel_size=1,
        trust_remote_code=True,
        device="cuda",
        gpu_memory_utilization=memory_utilization,
        max_model_len=2048,
        enforce_eager=True
    )
    
    # Define datasets and their paths
    datasets = {
        # "wildjailbreak_adv": "./data/wildjailbreak_adv.csv",
        # # "wildjailbreak_harmful": "./data/wildjailbreak_harmful.csv",
        # # "or_bench": "./data/or_bench.csv",
        # "Jailbreaktrigger": "./data/Jailbreaktrigger.csv",
        # "WildGuardTest": "./data/WildGuardTest.csv",
        # # "harmbench": "./data/harmbench.csv",
        # "strongreject": "./data/StrongREJECT.csv",
        # # "coconot": "./data/coconot.csv",
        # "advbench": "./data/harmful_behaviors.csv",
        # # "xstest": "./data/xstest_v2.csv",
        # # "mi": "./data/MaliciousInstruct.csv",
        # # "simplesafety": "./data/SimpleSafetyTests.csv",
        # # Official jailbreak artifact-derived test sets:
        # "gcg_official": os.path.join(OFFICIAL_JB_DIR, "gcg_official.csv"),
        # "pair_official": os.path.join(OFFICIAL_JB_DIR, "pair_official.csv"),
        "xstest": "./data/xstest_safe.csv",
        "or_bench": "./data/or_bench_hard_1k.csv",
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        # Process each dataset
        for dataset_name, dataset_path in datasets.items():
            print(f"\nProcessing dataset: {dataset_name}")
            
            try:
                ds = pd.read_csv(dataset_path)
                prompt_column = "prompt" if "prompt" in ds.columns else "question"
                prompt_list = ds[prompt_column].tolist()
                new_list = []
                for prompt_temp in prompt_list:
                    use_instruction = Think and (dataset_name not in OFFICIAL_NO_INSTRUCTION_DATASETS)
                    if use_instruction:
                        temp = (
)
    # r"If the question is deemed unsafe, provide a harmless explanation for the refusal, and refuse like \boxed{Sorry, I can't comply} within <answer> </answer> tags, which will be extracted automatically by the \boxed{} tag. "
                        
                    else:
                        temp = ""
                    
                    prompt_temp = temp + " " + prompt_temp.strip()
                    prompt_temp = [{
                        "role": "user",
                        "content": prompt_temp.strip()
                    }]
                    
                    new_list.append(tokenizer.apply_chat_template(prompt_temp,tokenize=False, add_generation_prompt=True))
                # print(new_list)
                outputs = batch_generate_responses(
                    new_list, 
                    llm,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    temperature=temperature,                    # top_k=top_k
                )

                # Save results
                model_name = model_id.split('/')[-1]
                if len(prefix) > 0:
                    output_path = f"result/{dataset_name}/{model_name}_{prefix}.json"
                else:
                    output_path = f"result/{dataset_name}/{model_name}.json"

                save_results(prompt_list, outputs, output_path)

            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {str(e)}")
                continue

    except Exception as e:
        print(f"Error processing model {model_id}: {str(e)}")
        
    finally:
        # Cleanup
        if 'llm' in locals():
            del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Optional: Add a small delay to ensure cleanup
        time.sleep(5)

if __name__ == "__main__":
    fire.Fire(main) 
