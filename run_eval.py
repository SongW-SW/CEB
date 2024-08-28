from src.task import fairness
from src.utils import file_process
from src.config import config
import argparse


# Load config, need to be re-written by using the argparser

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stereotype_recognition_data_json_path', type=str,
                        help='Path to the stereotype recognition data json file')
    args = parser.parse_args()

    evaluator = fairness.FairnessEval()
    # Load file of the stereotype recognition data
    print("Load data file from path: ", args.stereotype_recognition_data_json_path)
    stereotype_recognition_data = file_process.load_json(args.stereotype_recognition_data_json_path)
    print(evaluator.stereotype_recognition_eval(stereotype_recognition_data))

if __name__ == "__main__":
    eval()