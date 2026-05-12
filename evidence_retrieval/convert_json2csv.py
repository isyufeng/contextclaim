import json
import csv
import sys
from pathlib import Path

def process_generated_context(context):
    if context == "No relevant context found.":
        return ""

    lines = [line.strip() for line in context.split('\n') if line.strip()]

    if not lines:
        return ""

    def preprocess_line(line):
        import re
        line = re.sub(r'\(Relevance Score: \d+(\.\d+)?\)', '', line)
        line = re.sub(r'\(Word Count: \d+\)', '', line)
        line = re.sub(r'(?i)(\w+\s+)?summary:', '', line)
        return line.strip()

    if len(lines) == 1:
        return preprocess_line(lines[0])

    last_line = lines[-1]
    last_line_processed = preprocess_line(last_line)

    if len(last_line_processed.split()) >= 50:
        return last_line_processed
    else:
        max_words_line = max(lines, key=lambda x: len(preprocess_line(x).split()))
        return preprocess_line(max_words_line)


def convert_json_to_csv(input_file, output_file):
    """
    Convert a JSON file containing tweet data to a CSV file.

    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output CSV file
    """
    try:
        # Read JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Open CSV file for writing
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            # Define CSV writer and header
            csv_writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["tweet_id", "tweet_text", "class_label", "evidence"])

            # Check if the data is a list or a single object
            if isinstance(data, list):
                # Process each item in the list
                for item in data:
                    csv_writer.writerow([
                        item.get("tweet_id", ""),
                        item.get("tweet_text", ""),
                        item.get("class_label", ""),
                        # item.get("generated_context", "")
                        process_generated_context(item.get("generated_context", ""))
                    ])
            else:
                # Process a single JSON object
                csv_writer.writerow([
                    data.get("tweet_id", ""),
                    data.get("tweet_text", ""),
                    data.get("class_label", ""),
                    data.get("generated_context", "")
                ])

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if command line arguments are provided
    # if len(sys.argv) != 3:
    #     print("Usage: python json_to_csv.py <input_json_file> <output_csv_file>")
    #     sys.exit(1)

    # input_file = sys.argv[1]
    # output_file = sys.argv[2]

    input_dir = "data/evidence/CT22_claim/taslp"
    # input_dir = "data/PoliClaim/evidence"
    # input_dir = "data/CT22-Mistral"
    # input_dir = "data/CT22-GPT4o"
    input_dir = Path(input_dir)
    for file_path in input_dir.glob("*.json"):
        filename = file_path.name
        print(f"Processing file: {filename}")
        input_file = str(file_path)
        # output_file = input_file.replace(".json", ".csv")
        output_file = "data/evidence/CT22_claim/taslp/" + filename.replace(".json", ".csv")
        # output_file = "data/PoliClaim/evidence/" + filename.replace(".json", ".csv")
        # output_file = "data/CT22-GPT4o/" + filename.replace(".json", ".csv")
        convert_json_to_csv(input_file, output_file)
        print(f"Successfully converted {input_file} to {output_file}")
        # if "gpt4o" in filename and "interim" not in filename:
        #     print(f"Processing file: {filename}")
        #     input_file = str(file_path)
        #     # output_file = input_file.replace(".json", ".csv")
        #     output_file = "data/context_checkworthy/" + filename.replace(".json", ".csv")
        #     # output_file = "data/evidence/" + filename.replace(".json", ".csv")
        #     convert_json_to_csv(input_file, output_file)
        #     print(f"Successfully converted {input_file} to {output_file}")
        # else:
        #     print(f"Skip to process file: {filename}")


