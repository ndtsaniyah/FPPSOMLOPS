import json
import os
from pyabsa import AspectTermExtraction as ATEPC, available_checkpoints

# you can view all available checkpoints by calling available_checkpoints()
checkpoint_map = available_checkpoints()

aspect_extractor = ATEPC.AspectExtractor('multilingual',
                                         auto_device=True,  # False means load model on CPU
                                         cal_perplexity=True,
                                         )

def label_text(text):
    # Implementasi pelabelan dengan model yang sudah dilatih
    result = aspect_extractor.predict([text],
                         save_result=False,
                         print_result=True,  # print the result
                         ignore_error=True,  # ignore the error when the model cannot predict the input
                         )

    # Load existing results from JSON file if it exists
    output_file = "result.json"
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
    else:
        existing_results = []

    # Append new result to existing results
    existing_results.append(result)

    # Save the updated results back to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=4)
    return result
