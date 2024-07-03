import time
import torch
from fastchat.model import load_model, get_conversation_template
from src.utils.generation_utils import *
from dotenv import load_dotenv
import os
import json
import threading
from tqdm import tqdm
import urllib3
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LLMGeneration:
    def __init__(self,
                 test_type,
                 data_path,
                 dataset_name,
                 model_path,
                 online_model=False,
                 use_deepinfra=False,
                 use_replicate=False,
                 use_vllm=False,
                 repetition_penalty=1.0,
                 num_gpus=1,
                 max_new_tokens=512,
                 debug=False,
                 ):
        self.model_name = ""
        self.model_path = model_path        # model path. If using huggingface model, it should be the model path. Otherwise, it should be the model name.
        self.test_type = test_type          # test type, e.g., "stereotype_recognition"
        self.data_path = data_path          # path to the dataset
        self.dataset_name = dataset_name    # the dataset name, e.g., "winobias"
        self.online_model = online_model
        self.temperature = 0                                            # temperature setting for text generation, default 0.0 (greedy decoding)
        self.repetition_penalty = repetition_penalty                    # repetition penalty, default is 1.0
        self.num_gpus = num_gpus
        self.max_new_tokens = max_new_tokens                            # Number of max new tokens generated
        self.debug = debug
        self.online_model_list = get_models()[1]                        # Online model list, typically contains models that are not huggingface models
        self.model_mapping = get_models()[0]                            # Mapping between model path and model name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_replicate = use_replicate                              # Temporarily set to False as we don't use replicate api
        self.use_deepinfra = use_deepinfra                              # Temporarily set to False as we don't use deepinfra api
        self.use_vllm = use_vllm                                        # Set this to be True when using vLLM to run huggingface models
        self.model_name = model_mapping.get(self.model_path, "")        # Get the model name according to the model path

        if self.use_vllm:
            print("Using VLLM model for generation. Load model from: ", self.model_path)
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=num_gpus,
            )
            self.sampling_params = SamplingParams(temperature=self.temperature,
                                                  top_p=1.0, seed=1,
                                                  max_tokens=self.max_new_tokens)

        print("Model name: ", self.model_name)
        print("Model path: ", self.model_path)
        print("self.online_model_list: ", self.online_model_list)

    def _generation_hf(self, prompt, tokenizer, model, temperature):
        """
            Generates a response using a Hugging Face model.

            :param prompt: The input text prompt for the model.
            :param tokenizer: The tokenizer associated with the model.
            :param model: The Hugging Face model used for text generation.
            :param temperature: The temperature setting for text generation.
            :return: The generated text as a string.
        """

        prompt = prompt2conversation(self.model_path, prompt)
        inputs = tokenizer([prompt])
        inputs = {k: torch.tensor(v).to(self.device) for k, v in inputs.items()}
        output_ids = model.generate(
            **inputs,
            do_sample=True if temperature > 1e-5 else False,
            temperature=temperature,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
        )
        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]):]
        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs

    def _generation_vllm(self, prompt, tokenizer):
        """
            Generates a response using a VLLM model.
        """
        # TODO: Implement VLLM generation, which is faster and simpler than HF generation
        prompt = prompt2conversation_hf(prompt, tokenizer)
        response = self.llm.generate(prompt, self.sampling_params)
        return response[0].outputs[0].text

    def generation(self, model_name, prompt, tokenizer, model, temperature=None):
        """
            Generates a response using either an online or a local model.

            :param model_name: The name of the model.
            :param prompt: The input text prompt for the model.
            :param tokenizer: The tokenizer for the model.
            :param model: The model used for text generation.
            :param temperature: The temperature setting for text generation. Default is None.
            :return: The generated text as a string.
            """

        try:
            if (model_name in self.online_model_list) and self.online_model:
                # Using online models without using replicate or deepinfra apis
                ans = gen_online(model_name,
                                 prompt, temperature,
                                 replicate=self.use_replicate,
                                 deepinfra=self.use_deepinfra)
            elif (model_name in self.online_model_list) and ((self.online_model and self.use_replicate) or (self.online_model and self.use_deepinfra)):
                # Using online models with replicate or deepinfra apis
                ans = gen_online(model_name,
                                 prompt, temperature,
                                 replicate=self.use_replicate,
                                 deepinfra=self.use_deepinfra)
            elif self.use_vllm:
                ans = self._generation_vllm(prompt, tokenizer)
            else:
                ans = self._generation_hf(prompt, tokenizer, model, temperature)
            if not ans:
                raise ValueError("The response is NULL or an empty string!")
            return ans
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)

    def process_element(self, el, model, model_name, tokenizer, index, temperature, key_name='prompt'):
        """
            Processes a single element (data point) using the specified model.

            :param el: A dictionary containing the data to be processed.
            :param model: The model to use for processing.
            :param model_name: The name of the model.
            :param tokenizer: The tokenizer for the model.
            :param index: The index of the element in the dataset.
            :param temperature: The temperature setting for generation.
            :param key_name: The key in the dictionary where the prompt is located.
            """

        try:
            # If 'res' key doesn't exist or its value is empty, generate a new response
            if "res" not in el or not el['res']:
                res = self.generation(model_name=model_name,
                                      prompt=el[key_name],
                                      tokenizer=tokenizer,
                                      model=model,
                                      temperature=temperature)
                el['res'] = res
        except Exception as e:
            # Print error message if there's an issue during processing
            print(f"Error processing element at index {index}: {e}")

    def process_file(self, data_path, save_path, model_name, tokenizer, model, file_config, key_name='prompt'):
        """
            Processes a file containing multiple data points for text generation.

            :param data_path: Path to the input data file.
            :param save_path: Path where the processed data will be saved.
            :param model_name: The name of the model used for processing.
            :param tokenizer: The tokenizer for the model.
            :param model: The model to use for processing.
            :param file_config: Configuration settings for file processing.
            :param key_name: The key in the dictionary where the prompt is located.
            """
        if os.path.basename(data_path) not in file_config:
            print(f"{os.path.basename(data_path)} not in file_config")
            return

        with open(data_path) as f:
            print("Load data from {}".format(f.name))
            original_data = json.load(f)

        if os.path.exists(save_path):
            print(f"Load existing saved data from {save_path}")
            with open(save_path, 'r') as f:
                saved_data = json.load(f)
        else:
            saved_data = original_data

        GROUP_SIZE = 8 if self.online_model else 1
        for i in tqdm(range(0, len(saved_data), GROUP_SIZE), desc=f"Processing {data_path}", leave=False):
            group_data = saved_data[i:i + GROUP_SIZE]
            threads = []
            for idx, el in enumerate(group_data):
                temperature = file_config.get(os.path.basename(data_path), 0.0)
                t = threading.Thread(target=self.process_element,
                                     args=(el, model, model_name, tokenizer, idx, temperature, key_name))
                t.start()
                threads.append(t)
            file_process.save_json(saved_data, f"{save_path}")

            # Wait for all threads to complete
            for t in threads:
                t.join()
        print(f"Processed {data_path} and saved results to {save_path}")
        file_process.save_json(saved_data, f"{save_path}")

    def _run_task(self, model_name, model, tokenizer, base_dir, file_config, key_name='prompt'):
        """
            Runs a specific evaluation task based on provided parameters.

            :param model_name: The name of the model.
            :param model: The model used for processing.
            :param tokenizer: The tokenizer for the model.
            :param base_dir: Base directory containing test data files.
            :param file_config: Configuration settings for file processing.
            :param key_name: The key in the dictionary where the prompt is located.
            """

        test_res_dir = os.path.join(base_dir, 'test_res', model_name)
        if not os.path.exists(test_res_dir):
            os.makedirs(test_res_dir)
        section = base_dir.split('/')[-1]

        os.makedirs(os.path.join('generation_results', model_name, section), exist_ok=True)

        file_list = os.listdir(base_dir)
        for file in tqdm(file_list, desc="Processing files"):
            data_path = os.path.join(base_dir, file)
            save_path = os.path.join('generation_results', model_name, section, file)
            self.process_file(data_path, save_path, model_name, tokenizer, model, file_config, key_name)

    def run_fairness(self, model_name, model, tokenizer):
        base_dir = os.path.join(self.data_path, 'fairness')
        file_config = {
            "disparagement.json": 1.0,
            "preference.json": 1.0,
            "stereotype_agreement.json": 1.0,
            'stereotype_query_test.json': 1.0,
            'stereotype_recognition.json': 0.0,
        }
        self._run_task(model_name, model, tokenizer, base_dir, file_config)

    def run_stereotype_recognition(self, model_name, model, tokenizer):
        # TODO: Change the dataset path to the correct and well-organized path
        bbq_adapted = True
        base_dir = os.path.join(self.data_path, 'stereotype_recognition')
        if bbq_adapted:
            base_dir = os.path.join(base_dir, 'BBQ_stereo_recognition')
        # Name of the dataset file and the corresponding temperature setting
        # file_config = {
        #     self.dataset_name+'.json': 0.0,
        # }
        if bbq_adapted:
            file_config = {
                'bbq_age.json': 0.0,
                'bbq_gender.json': 0.0,
                'bbq_race.json': 0.0,
                'bbq_religion.json': 0.0,
            }
        else:
            file_config = {
                'winobias.json': 0.0,
                'crowspairs.json': 0.0,
                'redditbias.json': 0.0,
                'stereoset.json': 0.0,
            }
        # print("Load data from {}".format(os.path.join(base_dir, self.dataset_name+'.json')))

        self._run_task(model_name, model, tokenizer, base_dir, file_config)

    def run_stereotype_selection(self, model_name, model, tokenizer):
        bbq_adapted = True
        base_dir = os.path.join(self.data_path, 'stereotype_selection')
        if bbq_adapted:
            base_dir = os.path.join(base_dir, 'BBQ_stereo_selection')
        # Name of the dataset file and the corresponding temperature setting
        # file_config = {
        #     self.dataset_name+'.json': 0.0,
        # }
        if bbq_adapted:
            file_config = {
                'bbq_age.json': 0.0,
                'bbq_gender.json': 0.0,
                'bbq_race.json': 0.0,
                'bbq_religion.json': 0.0,
            }
        else:
            # key: dataset file name, value: temperature value
            file_config = {
                'winobias.json': 0.0,
                'crowspairs.json': 0.0,
                'redditbias.json': 0.0,
                'stereoset.json': 0.0,
            }
        # print("Load data from {}".format(os.path.join(base_dir, self.dataset_name+'.json')))

        self._run_task(model_name, model, tokenizer, base_dir, file_config)

    def run_toxicity_recognition(self, model_name, model, tokenizer):
        bbq_adapted = True
        base_dir = os.path.join(self.data_path, 'toxicity_recognition')
        if bbq_adapted:
            base_dir = os.path.join(base_dir, 'BBQ_toxic_recognition')
        # Name of the dataset file and the corresponding temperature setting
        # file_config = {
        #     self.dataset_name+'.json': 0.0,
        # }
        if bbq_adapted:
            file_config = {
                'bbq_age.json': 0.0,
                'bbq_gender.json': 0.0,
                'bbq_race.json': 0.0,
                'bbq_religion.json': 0.0,
            }
        else:
            # file_config = {
            #     'winobias.json': 0.0,
            #     'crowspairs.json': 0.0,
            #     'redditbias.json': 0.0,
            #     'stereoset.json': 0.0,
            # }
            raise ValueError("No file config defined for toxicity_recognition() when bbq_adapted is False")
        # print("Load data from {}".format(os.path.join(base_dir, self.dataset_name+'.json')))

        self._run_task(model_name, model, tokenizer, base_dir, file_config)

    def run_toxicity_selection(self, model_name, model, tokenizer):
        bbq_adapted = True
        base_dir = os.path.join(self.data_path, 'toxicity_selection')
        if bbq_adapted:
            base_dir = os.path.join(base_dir, 'BBQ_toxic_selection')
        # Name of the dataset file and the corresponding temperature setting
        # file_config = {
        #     self.dataset_name+'.json': 0.0,
        # }
        if bbq_adapted:
            file_config = {
                'bbq_age.json': 0.0,
                'bbq_gender.json': 0.0,
                'bbq_race.json': 0.0,
                'bbq_religion.json': 0.0,
            }
        else:
            # file_config = {
            #     'winobias.json': 0.0,
            #     'crowspairs.json': 0.0,
            #     'redditbias.json': 0.0,
            #     'stereoset.json': 0.0,
            # }
            raise ValueError("No file config defined for toxicity_selection() when bbq_adapted is False")
        # print("Load data from {}".format(os.path.join(base_dir, self.dataset_name+'.json')))

        self._run_task(model_name, model, tokenizer, base_dir, file_config)

    def run_stereotype_continuation(self, model_name, model, tokenizer):
        holistic_bias_adapted = True
        base_dir = os.path.join(self.data_path, 'stereotype_continuation')
        if holistic_bias_adapted:
            base_dir = os.path.join(base_dir, 'HolisticBias_stereo_continuation')
        # Name of the dataset file and the corresponding temperature setting
        # file_config = {
        #     self.dataset_name+'.json': 0.0,
        # }
        if holistic_bias_adapted:
            file_config = {
                'holisticbias_age.json': 0.8,
                'holisticbias_gender.json': 0.8,
                'holisticbias_race.json': 0.8,
                'holisticbias_religion.json': 0.8,
            }
        else:
            raise ValueError("No file config defined for stereotype_continuation() when holistic_bias_adapted is False")

        self._run_task(model_name, model, tokenizer, base_dir, file_config)

    def run_stereotype_conversation(self, model_name, model, tokenizer):
        holistic_bias_adapted = True
        base_dir = os.path.join(self.data_path, 'stereotype_conversation')
        if holistic_bias_adapted:
            base_dir = os.path.join(base_dir, 'HolisticBias_stereo_conversation')
        
        # Name of the dataset file and the corresponding temperature setting
        # file_config = {
        #     self.dataset_name+'.json': 0.0,
        # }
        if holistic_bias_adapted:
            file_config = {
                'holisticbias_age.json': 0.8,
                'holisticbias_gender.json': 0.8,
                'holisticbias_race.json': 0.8,
                'holisticbias_religion.json': 0.8,
            }
        else:
            raise ValueError("No file config defined for stereotype_conversation() when holistic_bias_adapted is False")

        self._run_task(model_name, model, tokenizer, base_dir, file_config)

    def run_toxicity_continuation(self, model_name, model, tokenizer):
        holistic_bias_adapted = True
        base_dir = os.path.join(self.data_path, 'toxicity_continuation')
        if holistic_bias_adapted:
            base_dir = os.path.join(base_dir, 'HolisticBias_toxic_continuation')
        # Name of the dataset file and the corresponding temperature setting
        # file_config = {
        #     self.dataset_name+'.json': 0.0,
        # }
        if holistic_bias_adapted:
            file_config = {
                'holisticbias_age.json': 0.8,
                'holisticbias_gender.json': 0.8,
                'holisticbias_race.json': 0.8,
                'holisticbias_religion.json': 0.8,
            }
        else:
            raise ValueError("No file config defined for toxicity_continuation() when holistic_bias_adapted is False")

        self._run_task(model_name, model, tokenizer, base_dir, file_config)

    def run_toxicity_conversation(self, model_name, model, tokenizer):
        holistic_bias_adapted = True
        base_dir = os.path.join(self.data_path, 'toxicity_conversation')
        if holistic_bias_adapted:
            base_dir = os.path.join(base_dir, 'HolisticBias_toxic_conversation')
        # Name of the dataset file and the corresponding temperature setting
        # file_config = {
        #     self.dataset_name+'.json': 0.0,
        # }
        if holistic_bias_adapted:
            file_config = {
                'holisticbias_age.json': 0.8,
                'holisticbias_gender.json': 0.8,
                'holisticbias_race.json': 0.8,
                'holisticbias_religion.json': 0.8,
            }
        else:
            raise ValueError("No file config defined for toxicity_conversation() when holistic_bias_adapted is False")

        self._run_task(model_name, model, tokenizer, base_dir, file_config)

    def run_decision_making(self, model_name, model, tokenizer):
        base_dir = os.path.join(self.data_path, 'decision_making')
        file_config = {
                'adult_gender.json': 0,
                'adult_gender_cf.json': 0,
                'adult_race.json': 0,
                'adult_race_cf.json': 0,
                'credit_age.json': 0,
                'credit_age_cf.json': 0,
                'credit_gender.json': 0,
                'credit_gender_cf.json': 0,
                'jigsaw_gender.json': 0,
                'jigsaw_race.json': 0,
                'jigsaw_religion.json': 0,
            }
        self._run_task(model_name, model, tokenizer, base_dir, file_config)


    def _run_single_test(self):
        """
            Executes a single test based on specified parameters.

            :param args: Contains parameters like test type, model name, and other configurations.
            :return: "OK" if successful, None otherwise.
            """
        model_name = self.model_name
        print(f"Beginning generation with {self.test_type} evaluation at temperature {self.temperature}.")
        print(f"Evaluation target model: {model_name}")
        if (model_name in self.online_model_list) and self.online_model:
            # Using online models without using replicate or deepinfra apis
            model, tokenizer = (None, None)
        elif (model_name in self.online_model_list) and ((self.online_model and self.use_replicate) or (self.online_model and self.use_deepinfra)):
            # Using online models with replicate or deepinfra apis
            model, tokenizer = (None, None)
        elif self.use_vllm:
            model = None
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("Using VLLM model for generation. Load model from: ", self.model_path)
        else:
            model, tokenizer = load_model(
            self.model_path,
            num_gpus=self.num_gpus,
            device=self.device,
            debug=self.debug,
        )

        test_functions = {
            'fairness': self.run_fairness,
            'stereotype_recognition': self.run_stereotype_recognition,
            'stereotype_selection': self.run_stereotype_selection,
            'toxicity_recognition': self.run_toxicity_recognition,
            'toxicity_selection': self.run_toxicity_selection,
            'stereotype_continuation': self.run_stereotype_continuation,
            'stereotype_conversation': self.run_stereotype_conversation,
            'toxicity_continuation': self.run_toxicity_continuation,
            'toxicity_conversation': self.run_toxicity_conversation,
            'decision_making': self.run_decision_making,
        }

        test_func = test_functions.get(self.test_type)
        if test_func:
            print(f"Running {self.test_type} test...")
            test_func(model_name=model_name, model=model, tokenizer=tokenizer)
            return "OK"
        else:
            print("Invalid test_type. Please provide a valid test_type.")
            return None

    def generation_results(self, max_retries=2, retry_interval=3):
        """
            Main function to orchestrate the test runs with retries.

            :param args: Command-line arguments for the test run.
            :param max_retries: Maximum attempts to run the test.
            :param retry_interval: Time interval between retries in seconds.
            :return: Final state of the test run.
            """
        if not os.path.exists(self.data_path):
            print(f"Dataset path {self.data_path} does not exist.")
            return None

        
        for attempt in range(max_retries):
            try:
                state = self._run_single_test()
                if state:
                    print(f"Test function successful on attempt {attempt + 1}")
                    return state
            except Exception as e:
                print(f"Test function failed on attempt {attempt + 1}: {e}")
                print(f"Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)

        print("Test failed after maximum retries.")
        return None
