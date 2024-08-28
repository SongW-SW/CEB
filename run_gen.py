from src.generation.generation import LLMGeneration
import argparse


def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--test_type', type=str, help='Type of test')
    parser.add_argument('--online_model', action='store_true', help='Use online model')
    parser.add_argument('--use_vllm', type=bool, help='Use VLLM model')
    args = parser.parse_args()

    if args.online_model:
        print("Using online model")
        #### ChatGPT (GPT-3.5-Turbo) and GPT-4-Turbo generation ####
        llm_gen = LLMGeneration(
            model_path=args.model_path, 
            test_type=args.test_type, 
            data_path="./data/",
            dataset_name=None,          # run on all datasets in the folder
            online_model=True, 
            use_deepinfra=False,
            use_replicate=False,
            use_vllm=False,
            repetition_penalty=1.0,
            num_gpus=1, 
            max_new_tokens=512, 
            debug=False
        )
    
    else:
        print("Using vLLM model")
        #### Using local models such as Llama2 and Llama3 to generate ####
        llm_gen = LLMGeneration(
            model_path=args.model_path, 
            test_type=args.test_type, 
            data_path="./data/",
            dataset_name=None,          # run on all datasets in the folder
            online_model=False, 
            use_deepinfra=False,
            use_replicate=False,
            use_vllm=True,
            repetition_penalty=1.0,
            num_gpus=1, 
            max_new_tokens=512, 
            debug=False
        )

    llm_gen.generation_results()

if __name__ == "__main__":
    generate()