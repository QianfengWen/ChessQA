#!/usr/bin/env python3
"""
Script to format JSONL results file into a more readable format.
"""

import json
import sys
from pathlib import Path

def format_jsonl(input_file, output_file=None):
    """Format a JSONL file to be more readable."""

    # If no output file specified, create one with _formatted suffix
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_formatted.json"

    results = []

    # Read the JSONL file
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # Create a summary statistics
    total_tasks = len(results)
    correct = sum(1 for r in results if r.get('is_correct', False))

    # Group by task type
    task_types = {}
    for result in results:
        task_type = result.get('task_type', 'unknown')
        if task_type not in task_types:
            task_types[task_type] = {'correct': 0, 'total': 0, 'tasks': []}

        task_types[task_type]['total'] += 1
        if result.get('is_correct', False):
            task_types[task_type]['correct'] += 1

        # Create a simplified view
        simplified = {
            'task_id': result['task_id'],
            'correct': result.get('is_correct', False),
            'expected': result.get('correct_answer', 'N/A'),
            'predicted': result.get('extracted', 'N/A'),
            'error_type': result.get('error_type', 'N/A')
        }
        task_types[task_type]['tasks'].append(simplified)

    # Create the formatted output
    output = {
        'summary': {
            'total_tasks': total_tasks,
            'correct': correct,
            'accuracy': f"{(correct/total_tasks)*100:.2f}%" if total_tasks > 0 else "0%"
        },
        'by_task_type': {}
    }

    # Add task type summaries
    for task_type, data in sorted(task_types.items()):
        accuracy = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
        output['by_task_type'][task_type] = {
            'summary': {
                'total': data['total'],
                'correct': data['correct'],
                'accuracy': f"{accuracy:.2f}%"
            },
            'tasks': data['tasks']
        }

    # Write the formatted output
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Formatted results written to: {output_file}")
    print(f"\nSummary:")
    print(f"  Total Tasks: {total_tasks}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {(correct/total_tasks)*100:.2f}%\n")

    print("By Task Type:")
    for task_type, data in sorted(task_types.items()):
        accuracy = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
        print(f"  {task_type}: {data['correct']}/{data['total']} ({accuracy:.2f}%)")

    # Also create a human-readable text report
    report_file = Path(input_file).parent / f"{Path(input_file).stem}_report.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CHESS LLM BENCHMARK RESULTS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Model: {Path(input_file).stem}\n")
        f.write(f"Total Tasks: {total_tasks}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Overall Accuracy: {(correct/total_tasks)*100:.2f}%\n\n")

        f.write("-"*80 + "\n")
        f.write("PERFORMANCE BY TASK TYPE\n")
        f.write("-"*80 + "\n\n")

        for task_type, data in sorted(task_types.items()):
            accuracy = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
            f.write(f"{task_type}:\n")
            f.write(f"  Correct: {data['correct']}/{data['total']}\n")
            f.write(f"  Accuracy: {accuracy:.2f}%\n\n")

        f.write("-"*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("-"*80 + "\n\n")

        for task_type, data in sorted(task_types.items()):
            f.write(f"\n{task_type.upper()}\n")
            f.write("="*40 + "\n")
            for task in data['tasks'][:5]:  # Show first 5 of each type
                f.write(f"  Task ID: {task['task_id']}\n")
                f.write(f"    Result: {'✓ CORRECT' if task['correct'] else '✗ INCORRECT'}\n")
                if not task['correct']:
                    f.write(f"    Expected: {task['expected']}\n")
                    f.write(f"    Predicted: {task['predicted']}\n")
                    f.write(f"    Error Type: {task['error_type']}\n")
                f.write("\n")

            if data['total'] > 5:
                f.write(f"  ... and {data['total'] - 5} more tasks\n\n")

    print(f"\nDetailed report written to: {report_file}")

if __name__ == "__main__":
    input_file = "chess-llm-benchmark/results/anthropic_claude-3.5-haiku.jsonl"
    format_jsonl(input_file)