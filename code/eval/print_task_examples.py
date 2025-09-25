#!/usr/bin/env python3
"""Print one example per task type from benchmark files."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_tasks_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load tasks from a JSONL file."""
    tasks = []
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist")
        return tasks

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    task = json.loads(line)
                    tasks.append(task)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num} in {file_path}: {e}")

    return tasks


def print_task_example(task: Dict[str, Any], task_type: str):
    """Print a detailed example of a task."""
    print("=" * 80)
    print(f"TASK TYPE: {task_type}")
    print("=" * 80)

    # Print full JSON (formatted)
    print("\nFULL JSON:")
    print("-" * 40)
    print(json.dumps(task, indent=2, ensure_ascii=False))

    # Print the prompt separately for clarity
    print("\nPROMPT:")
    print("-" * 40)
    print(task.get('question', 'No question field found'))

    # Print key metadata
    print(f"\nKEY INFO:")
    print("-" * 40)
    print(f"Task ID: {task.get('task_id', 'N/A')}")
    print(f"Input FEN: {task.get('input', 'N/A')}")
    print(f"Correct Answer: {task.get('correct_answer', 'N/A')}")

    if 'metadata' in task and task['metadata']:
        print(f"Metadata keys: {list(task['metadata'].keys())}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Print one example per task type from benchmark files")

    default_benchmark_dir = Path(__file__).parent.parent.parent / "data" / "benchmark"

    parser.add_argument("--benchmark_dir", type=Path, default=default_benchmark_dir,
                       help="Directory containing benchmark JSONL files")
    parser.add_argument("--files", type=str, nargs="*", default=["structural.jsonl"],
                       help="Specific files to process (if not provided, process all .jsonl files)")
    parser.add_argument("--task_types", type=str, nargs="*",
                       help="Specific task types to show (if not provided, show one per type)")

    args = parser.parse_args()

    # Get list of files to process
    if args.files:
        files_to_process = [args.benchmark_dir / f for f in args.files]
    else:
        files_to_process = list(args.benchmark_dir.glob("*.jsonl"))

    if not files_to_process:
        print(f"No JSONL files found in {args.benchmark_dir}")
        return

    print(f"Processing {len(files_to_process)} files from {args.benchmark_dir}")
    print(f"Files: {[f.name for f in files_to_process]}")
    print()

    # Group tasks by task_type
    task_examples = defaultdict(list)

    for file_path in files_to_process:
        print(f"Loading tasks from {file_path.name}...")
        tasks = load_tasks_from_file(file_path)

        for task in tasks:
            task_type = task.get('task_type', 'unknown')
            task_examples[task_type].append(task)

    # Filter by requested task types if specified
    if args.task_types:
        filtered_examples = {k: v for k, v in task_examples.items() if k in args.task_types}
        task_examples = filtered_examples

    # Print summary
    print(f"\nFound {len(task_examples)} unique task types:")
    for task_type, tasks in sorted(task_examples.items()):
        print(f"  - {task_type}: {len(tasks)} tasks")
    print()

    # Print one example per task type (with special handling for puzzle tasks)
    puzzle_level_shown = False
    puzzle_theme_shown = False

    for task_type, tasks in sorted(task_examples.items()):
        if tasks:
            # Special handling for puzzle tasks - only show one level and one theme
            if task_type.startswith('puzzle_level_'):
                if puzzle_level_shown:
                    continue
                puzzle_level_shown = True
            elif task_type.startswith('puzzle_theme_'):
                if puzzle_theme_shown:
                    continue
                puzzle_theme_shown = True

            # Take the first task as example
            example_task = tasks[0]
            print_task_example(example_task, task_type)

    print(f"Showed examples for {len(task_examples)} task types")


if __name__ == "__main__":
    main()