"""Retrieval-Augmented (iterative search) QA runner.

Workflow:
  1. The model is prompted with instructions to use a <search> tag. If it returns <search>query</search>, an external search is performed (Serper).
  2. The top organic results are formatted and injected into a <information> block for the next turn.
  3. This loop continues until the model provides an <answer>...</answer> or the max iteration count is reached.
  4. Accuracy is calculated, and a detailed log (including reasoning and search steps) is saved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

################################
# CONFIG (User-configurable parameters)
################################
DEFAULT_DATA_PATH = ""
MODEL_NAME = ""  

# OpenAI (or compatible) API configuration
OPENAI_BASE_URL = "YOUR_API_BASE_URL"
OPENAI_API_KEY = "YOUR_API_KEY"

# Search (Serper) configuration
SERPER_API_KEY = "YOUR_SERPER_API_KEY"
SERPER_ENDPOINT = "https://google.serper.dev/search"
SERPER_TOP_DOCS = 3  # Number of top organic results to include

# Generation parameters
MAX_TOKENS = 4096
TEMPERATURE = 0.7
MAX_ITERATIONS = 10
MAX_THREADS = 4

# Exponential backoff for retries
RETRY_INITIAL_DELAY = 0.5
RETRY_MAX_DELAY = 10

################################
# Client Initialization
################################
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="QA Test Runner (Iterative Search / RAG)")
        parser.add_argument("data", type=str, nargs="?", default=DEFAULT_DATA_PATH, help="Path to the test data file")
        parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name to override the default")
        parser.add_argument("--threads", type=int, default=MAX_THREADS, help="Number of parallel threads")
        parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature")
        parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Maximum tokens to generate")
        parser.add_argument("--max-iter", type=int, default=MAX_ITERATIONS, help="Maximum search iterations")
        parser.add_argument("--serper-key", type=str, default=SERPER_API_KEY, help="Override for Serper API Key")
        return parser.parse_args()

def get_query(text: str) -> Optional[str]:
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else None

def search(query: str, *, serper_key: str) -> str:
    payload = json.dumps({"q": query})
    headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
    max_retries = 5
    results: Dict[str, Any] = {}
    for attempt in range(max_retries):
        try:
            response = requests.post(SERPER_ENDPOINT, headers=headers, data=payload, timeout=10)
            if response.status_code == 200:
                results = response.json()
                break
            print(f"Serper request failed (status {response.status_code}), attempt {attempt+1}/{max_retries}")
        except Exception as e:  # noqa: BLE001
            print(f"Serper request error: {e}, attempt {attempt+1}/{max_retries}")
        if attempt == max_retries - 1:
            results = {}

    out = []
    organic = results.get("organic", [])[:SERPER_TOP_DOCS]
    for idx, doc_item in enumerate(organic):
        title = doc_item.get("title", "")
        snippet = doc_item.get("snippet", "")
        out.append(f"Doc {idx+1}(Title: {title}) {snippet}")
    return "\n".join(out)

def simple_match(predicted: str, expected: str) -> bool:
    predicted = predicted.lower().strip()
    expected = expected.lower().strip()
    return expected in predicted

def call_model(messages: List[Dict[str, Any]], *, model_name: str, max_tokens: int, temperature: float) -> Any:
    retry_count = 0
    delay = RETRY_INITIAL_DELAY
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            retry_count += 1
            print(f"Error calling {model_name} (attempt {retry_count}): {e}")
            print(f"Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            delay = min(delay * 2, RETRY_MAX_DELAY)

results: List[Dict[str, Any]] = []
results_lock = threading.Lock()
correct_count = 0
correct_count_lock = threading.Lock()
total_count = 0  # Assigned at runtime


def process_qa_item(item_data, *, model_name: str, temperature: float, max_tokens: int, max_iterations: int, serper_key: str, total: int):
    idx, item = item_data
    question = item["question"]
    expected_answer = item["answer"]
    if not question.endswith("?"):
        question += "?"

    initial_prompt = (
        "Answer the given question. You must conduct reasoning inside <think> and </think> every time you get new information. "
        "If you lack knowledge, call search by <search> query </search>; you will receive results enclosed by <information> </information>. "
        "Search as many times as needed. When sufficient knowledge is gathered, output ONLY the final answer inside <answer> </answer> without extra explanation. "
        f"For example, <answer> Beijing </answer>. Question: {question}"
    )

    print(f"\n=== Question {idx + 1}/{total} ===")
    print(f"Question: {question}")
    print(f"Expected: {expected_answer}")

    messages: List[Dict[str, Any]] = [{"role": "user", "content": initial_prompt}]
    reasoning_process: List[Dict[str, Any]] = [{"type": "initial_prompt", "content": initial_prompt}]

    iteration = 0
    search_count = 0
    ans = ""

    while iteration < max_iterations:
        response_text = call_model(messages, model_name=model_name, max_tokens=max_tokens, temperature=temperature)
        if not response_text:
            ans = f"Error: Failed to get response from {model_name}"
            reasoning_process.append({"type": "error", "step": iteration, "content": ans})
            break

        reasoning_process.append({"type": "model_output", "step": iteration, "content": response_text})
        search_query = get_query(response_text)

        if search_query:
            search_results = search(search_query, serper_key=serper_key)
            search_count += 1
            reasoning_process.append({"type": "search_query", "step": iteration, "query": search_query})
            reasoning_process.append({"type": "search_results", "step": iteration, "content": search_results})
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"<information>{search_results}</information>"})
            iteration += 1
            continue

        reasoning_process.append({"type": "final_output", "content": response_text})
        if "<answer>" in response_text and "</answer>" in response_text:
            ans = response_text[response_text.find("<answer>") + 8: response_text.find("</answer>")].strip()
        else:
            ans = response_text.strip()
        break

    if iteration >= max_iterations and not ans:
        ans = "Error: Maximum search iterations reached"
        reasoning_process.append({"type": "error", "content": ans})

    is_correct = simple_match(ans, expected_answer)
    print(f"Model Answer: {ans}")
    print(f"Correct: {'✓' if is_correct else '✗'}")

    result = {
        "question": question,
        "expected_answer": expected_answer,
        "model_answer": ans,
        "is_correct": is_correct,
        "search_count": search_count,
        "reasoning_process": reasoning_process,
    }
    with results_lock:
        results.append(result)
    if is_correct:
        with correct_count_lock:
            global correct_count
            correct_count += 1
    return result


def save_results(*, model_name: str, data_path: str, total: int, correct: int, accuracy: float):
    year_match = re.search(r"(\d{4})", data_path)
    year = year_match.group(1) if year_match else "unknown"
    level_match = re.search(r"(level\d+)", os.path.basename(data_path))
    level = level_match.group(1) if level_match else "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe_name = model_name.replace("-", "_").replace(".", "_").replace("/", "_")
    output_dir = os.path.join("output", year)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{level}_results_RAG_{model_safe_name}_search_{timestamp}.json")
    summary_file = os.path.join(output_dir, f"{level}_summary_RAG_{model_safe_name}_search_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    summary = {
        "model": f"{model_name} (Search enabled)",
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": accuracy,
        "test_file": data_path,
        "timestamp": timestamp,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Detailed results saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")


def main():
    args = parse_args()
    model_name = args.model
    threads = args.threads
    temperature = args.temperature
    max_tokens = args.max_tokens
    max_iter = args.max_iter
    serper_key = args.serper_key

    with open(args.data, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    global total_count
    total_count = len(test_data["qa_pairs"])
    print(
        f"Testing {total_count} QA pairs with {model_name} (Search enabled) "
        f"using {threads} parallel threads..."
    )

    with ThreadPoolExecutor(max_workers=threads) as executor:
        qa_items = [(idx, item) for idx, item in enumerate(test_data["qa_pairs"])]
        list(
            executor.map(
                lambda data: process_qa_item(
                    data,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_iterations=max_iter,
                    serper_key=serper_key,
                    total=total_count,
                ),
                qa_items,
            )
        )

    accuracy = correct_count / total_count if total_count else 0.0
    print(f"\n=== Final Results ({model_name} Search enabled) ===")
    print(f"Total questions: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    save_results(model_name=model_name, data_path=args.data, total=total_count, correct=correct_count, accuracy=accuracy)


if __name__ == "__main__":
    main()