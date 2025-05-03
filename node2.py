import re
import argparse

def remove_think_blocks(text: str) -> str:
    """
    Remove all <think>...</think> blocks (including the tags) from the input text.
    """
    # DOTALL so that '.*?' matches newlines
    pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
    return pattern.sub('', text)

def process_file(input_path: str, output_path: str):
    """
    Read the input file, strip out all <think> blocks, and write the result to output_path.
    """
    with open(input_path, 'r', encoding='utf-8') as infile:
        content = infile.read()

    cleaned = remove_think_blocks(content)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(cleaned)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove all <think>...</think> paragraphs from a text file"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input .txt file"
    )
    parser.add_argument(
        "-o", "--output",
        default="cleaned.txt",
        help="Path to write the cleaned output (default: cleaned.txt)"
    )
    args = parser.parse_args()

    process_file(args.input_file, args.output)
    print(f"✅ Cleaned file written to {args.output}")
