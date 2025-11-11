import pickle
import json

# LLM-MLLM Output
llm_out = "fc_detailed_results/qwen_qwen/16.pkl"
# Justification Output
just_out = "fc_detailed_results/qwen_qwen/16_justification.pkl"

# Eval Output
eval_out = "fc_detailed_results/qwen_qwen/12_justification.pkl"

# Path to your pickle file
file_path = llm_out

# Safely load the pickle file
with open(file_path, "rb") as file:
    data = pickle.load(file)

with open(just_out, "rb") as file:
    data_just = pickle.load(file)

# Display or use the loaded data
print(type(data))
# print(data)

with open(file_path.split('/')[-1] + '.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
print(file_path.split('/')[-1] + '.json' + ' is done...')