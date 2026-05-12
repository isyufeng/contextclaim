import os
import argparse
from pathlib import Path
import torch

from keyword_extractor import extract_keywords_from_file
from semantic_entity_linker import SemanticEntityLinker


def setup_directories(base_dir=None):
    if base_dir is None:
        base_dir = Path('data')
    else:
        base_dir = Path(base_dir)

    # Create directories
    dirs = {
        'keywords': base_dir / 'keywords',
        'linked_entities': base_dir / 'linked_entities',
        'cache': base_dir / 'cache'
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def process_file(input_file, output_dir=None, model_name='dslim/bert-base-NER',
                 batch_size=32, device=None):
    # Set up device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    dirs = setup_directories(output_dir)

    print(f"Step 1: Extracting keywords from {input_file}")
    keywords_path, keywords_df = extract_keywords_from_file(
        input_file,
        model_name=model_name,
        output_dir=str(dirs['keywords']),
        batch_size=batch_size,
        device=device
    )

    print(f"Step 2: Linking entities from {keywords_path}")
    linker = SemanticEntityLinker(
        cache_dir=str(dirs['cache']),
        model_name='all-MiniLM-L6-v2',
        device=device
    )

    entity_batch_size = min(batch_size * 2, 200)  # Entity linking can use larger batches
    entities_path = linker.process_json_file(
        keywords_path,
        output_dir=str(dirs['linked_entities']),
        batch_size=entity_batch_size
    )

    return keywords_path, entities_path


def process_directory(input_dir, output_dir=None, model_name='dslim/bert-base-NER',
                      file_pattern='*.tsv', batch_size=32, device=None):
    input_dir = Path(input_dir)
    processed_files = []

    for file_path in input_dir.glob(file_pattern):
        print(f"Processing file: {file_path}")
        keywords_path, entities_path = process_file(
            str(file_path),
            output_dir=output_dir,
            model_name=model_name,
            batch_size=batch_size,
            device=device
        )
        processed_files.append((file_path, keywords_path, entities_path))

    return processed_files


def main():
    parser = argparse.ArgumentParser(description="COVID-19 Tweet Analysis Pipeline with GPU Support")
    parser.add_argument('--input', required=True, help='Input file or directory')
    parser.add_argument('--output_dir', default='data', help='Output directory')
    parser.add_argument('--model', default='dslim/bert-base-NER', help='NER model name')
    parser.add_argument('--pattern', default='*.tsv', help='File pattern to match if input is a directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None,
                        help='Device to use (cuda/cpu); auto-detects if not specified')

    args = parser.parse_args()

    # Print GPU information if available
    print(f"Using device: {args.device}")

    input_path = Path(args.input)

    if input_path.is_file():
        # Process single file
        keywords_path, entities_path = process_file(
            str(input_path),
            output_dir=args.output_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device
        )
        print(f"\nProcessing complete:")
        print(f"  Input file: {input_path}")
        print(f"  Keywords extracted to: {keywords_path}")
        print(f"  Entities linked to: {entities_path}")

    elif input_path.is_dir():
        # Process directory
        processed_files = process_directory(
            str(input_path),
            output_dir=args.output_dir,
            model_name=args.model,
            file_pattern=args.pattern,
            batch_size=args.batch_size,
            device=args.device
        )

        print(f"\nProcessing complete:")
        print(f"  Input directory: {input_path}")
        print(f"  Files processed: {len(processed_files)}")
        for input_file, keywords_file, entities_file in processed_files:
            print(f"\n  - {input_file.name}")
            print(f"    Keywords extracted to: {keywords_file}")
            print(f"    Entities linked to: {entities_file}")

    else:
        print(f"Error: Input path '{input_path}' does not exist")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
