import json

# Load the model output from the JSON file
with open('./model_outputs_Mistral-7B-v0.1_5shots.json', 'r') as file:
    model_outputs_MIstral_7b = json.load(file)

# Initialize a dictionary to store the categorized data
categorized_output = {}

# Group the data by category
for entry in model_outputs_MIstral_7b:
    category = entry.get('category', 'Uncategorized')  # Use 'Uncategorized' if no category field is found
    
    if category not in categorized_output:
        categorized_output[category] = []  # Create a new list for the category if it doesn't exist
    
    # Add the entire entry under the respective category
    categorized_output[category].append({
        'question_id': entry.get('question_id'),
        'question': entry.get('question'),
        'options': entry.get('options'),
        'answer': entry.get('answer'),
        'answer_index': entry.get('answer_index'),
        'cot_content': entry.get('cot_content'),
        'src': entry.get('src'),
        'pred': entry.get('pred'),
        'generated_text': entry.get('generated_text')
    })

# Save the grouped data as a new JSON file for each category
for category, entries in categorized_output.items():
    # Define the output file name based on the category
    file_name = f"{category}.json"
    
    # Save the data for this category to a separate file
    with open(file_name, 'w') as json_file:
        json.dump(entries, json_file, indent=4)
    
    # Optionally, print the categorized output for verification
    print(f"Saved {file_name} with {len(entries)} entries.")

