import os
import difflib
import inspect
from src.task import fairness
from src.utils import file_process
import traceback


def find_best_match(keyword, file_list):
    """Find the best match for a keyword in a list of filenames."""
    matches = difflib.get_close_matches(keyword, file_list, n=1, cutoff=0.1)
    return matches[0] if matches else None

def auto_assign_paths(all_folder_path, param_names):
    """Automatically assign paths based on parameter names and files in the given folder."""
    files = os.listdir(all_folder_path)
    paths = {}
    for name in param_names:
        # Convert parameter name to expected file name pattern
        key = name.replace('_path', '')
        expected_filename = f"{key}.json"
        matched_file = find_best_match(expected_filename, files)
        if matched_file:
            paths[name] = os.path.join(all_folder_path, matched_file)
    return paths

def run_fairness(
    all_folder_path=None,
    stereotype_recognition_path=None,
    stereotype_agreement_path=None,
    stereotype_query_test_path=None,
    disparagement_path=None,
    preference_path=None,
):

    if all_folder_path:
        stereotype_recognition_path = os.path.join(all_folder_path, "stereotype_recognition.json")
        stereotype_agreement_path = os.path.join(all_folder_path, "stereotype_agreement.json")
        stereotype_query_test_path = os.path.join(all_folder_path, "stereotype_query_test.json")
        disparagement_path = os.path.join(all_folder_path, "disparagement.json")
        preference_path = os.path.join(all_folder_path, "preference.json")

    evaluator = fairness.FairnessEval()

    (
        stereotype_recognition_res,
        stereotype_agreement_res,
        stereotype_query_res,
        disparagement_res,
        preference_res,
    ) = (None, None, None, None, None)

    if stereotype_recognition_path is not None:
        stereotype_recognition_data = file_process.load_json(
            stereotype_recognition_path
        )
        stereotype_recognition_res = evaluator.stereotype_recognition_eval(
            stereotype_recognition_data
        )

    if stereotype_agreement_path is not None:
        stereotype_agreement_data = file_process.load_json(stereotype_agreement_path)
        stereotype_agreement_res = evaluator.stereotype_agreement_eval(
            stereotype_agreement_data
        )

    if stereotype_query_test_path is not None:
        stereotype_query_data = file_process.load_json(stereotype_query_test_path)
        stereotype_query_res = evaluator.stereotype_query_eval(stereotype_query_data)

    if disparagement_path is not None:
        disparagement_data = file_process.load_json(disparagement_path)
        disparagement_res = evaluator.disparagement_eval(disparagement_data)

    if preference_path is not None:
        preference_data = file_process.load_json(preference_path)
        preference_res = evaluator.preference_eval(preference_data)

    return {
        "stereotype_recognition": stereotype_recognition_res,
        "stereotype_agreement": stereotype_agreement_res,
        "stereotype_query": stereotype_query_res,
        "disparagement": disparagement_res,
        "preference": preference_res,
    }
