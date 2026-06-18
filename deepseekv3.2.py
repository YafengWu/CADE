import pandas as pd
import os
import json
import time
import re
from openai import OpenAI


# 1. Configuration
client = OpenAI(
    api_key=("your key"),
    base_url="https://api.siliconflow.cn/v1"
)

# 2. Define the Test Datasets
base_data_path = "/weka/s225635478/CADE/data/test"
test_files = {
    "Classification": "classification.csv",
    "Anomaly": "anomaly_detection.csv",
    "Forecasting": "forecasting.csv",
    "Imputation": "imputation.csv",
    "Multiple_choice": "multiple_choice.csv",
    "True_false": "true_false.csv"
}


def generate_answer(prompt_text, max_retries=5, task_name=""):
    """Generate answer using DeepSeek API with retry logic for rate limits."""
    
    task_prompts = {
        "Imputation": 'Please answer directly with the format <answer>[val1, val2, ...]</answer>. Return the complete time series with all missing values filled in.',
        "Forecasting": 'Please respond with ONLY the predicted values in a Python list format [v1, v2, ..., vN], where N equals the exact number of requested points. No explanations.',
        "Classification": 'Please answer directly with the format <answer>Label</answer>. Choose exactly one label from the provided options.',
        "Anomaly": 'Please answer directly with the format <answer>Normal</answer> or <answer>Anomaly</answer>.',
        "Multiple_choice": 'Please answer directly with the format <answer>Option</answer>. Choose exactly one option from the provided options.',
        "True_false": 'Please answer directly with the format <answer>True</answer> or <answer>False</answer>.',
    }
    
    suffix = task_prompts.get(task_name, "")
    if suffix:
        prompt_text = prompt_text + "\n\n" + suffix

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3.2",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "ResourceExhausted" in error_str:
                wait_time = 2 ** attempt * 5
                print(f"  Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                print(f"  API error: {e}")
                return f"Error: {e}"
    return "Error: Max retries exceeded due to rate limiting"


# 3. Helper function to parse QA_list
def parse_qa_list(qa_string):
    if not qa_string or pd.isna(qa_string):
        return None

    # Try clean JSON first
    for attempt in [qa_string.strip(), '{' + qa_string.strip() + '}']:
        try:
            return json.loads(attempt)
        except:
            pass

    # Fallback: extract question and answer via string matching
    try:
        s = qa_string.strip()
        # Find "question": "..." and "answer": "..."
        q_match = re.search(r'"question"\s*:\s*"', s)
        a_match = re.search(r'",\s*"answer"\s*:\s*"', s)
        
        if q_match and a_match:
            question = s[q_match.end():a_match.start()]
            # Find the answer value (everything after "answer": " until the trailing ")
            answer_start = a_match.end()
            answer = s[answer_start:].rstrip().rstrip('}').rstrip('"')
            return {"question": question, "answer": answer}
        elif q_match:
            # No answer field — extract question only
            question = s[q_match.end():].rstrip().rstrip('}').rstrip('"')
            return {"question": question}
    except:
        pass

    return None


# 4. Execution Loop
for task_name, file_name in test_files.items():
    file_path = os.path.join(base_data_path, file_name)

    if not os.path.exists(file_path):
        print(f"Skipping {task_name}: File not found.")
        continue

    print(f"\n>>> Running Task: {task_name}")
    df = pd.read_csv(file_path)

    predictions = []

    for index, row in df.iterrows():
        qa_list = row.get('QA_list')
        application_domain = row.get('application_domain', '')
        task_type = row.get('task_type', '')

        if not qa_list or pd.isna(qa_list):
            predictions.append("Error: No QA_list found")
            continue

        qa_dict = parse_qa_list(qa_list)

        if qa_dict is None:
            predictions.append("Error: Could not parse QA_list")
            print(f"Row {index}: Failed to parse: {qa_list[:100]}...")
            continue

        question = qa_dict.get('question', '')

        if not question:
            predictions.append("Error: No question found in QA_list")
            continue

        full_context = f"Application Domain: {application_domain}\nTask Type: {task_type}\n\n{question}"

        ans = generate_answer(full_context, task_name=task_name)
        predictions.append(ans)

        if index % 10 == 0:
            print(f"Processed {index}/{len(df)} rows...")

        # Small delay to avoid hitting rate limits
        time.sleep(1)

    # Save results
    df['model_response'] = predictions
    output_dir = "results/deepseekv3.2"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"results_{task_name.lower()}.csv")
    df.to_csv(output_filename, index=False)
    print(f"Saved results to {output_filename}")

print("\n--- All tasks completed successfully ---")