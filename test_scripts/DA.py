"""Direct Answer (DA) QA runner (no external search).

特点:
    - 直接回答模式: 不提供链式推理提示, 仅输出最终答案标签 <answer>...</answer>
    - 多线程并行
    - 简单包含式准确率统计
    - 结果 & 汇总 JSON 自动按年份/level/时间戳命名

新增:
    - 顶部集中 CONFIG, 方便批量修改
    - 支持将 `MODEL_NAME` 设为 "gpt-5" 或其它模型; 若启用带 "search-preview" 或 "deep-research" 的模型名, 将自动走特殊分支
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
from typing import Any, Dict, List

from openai import OpenAI

###############################
# CONFIG (用户可配置参数集中区)
###############################
# 数据文件默认路径
DEFAULT_DATA_PATH = "data/2025/level2.json"

# 模型名称: 可改为 "gpt-5" / "gpt-4o" / "gpt-4o-search-preview" / "o3-deep-research" / 自建模型别名
MODEL_NAME = "gpt-5"

# OpenAI (或兼容) 服务端点与密钥
OPENAI_BASE_URL = "YOUR_API_BASE_URL"  # 例如: https://api.openai.com/v1/
OPENAI_API_KEY = "YOUR_API_KEY"

# 推理生成控制
MAX_THREADS = 4          # 并行线程数
MAX_TOKENS = 2048        # 最多生成 token 数
TEMPERATURE = 0.7        # 采样温度

# 调用重试策略
RETRY_INITIAL_DELAY = 0.5
RETRY_MAX_DELAY = 10

# 若需要在命令行覆盖, 使用: python DA.py data/xxx.json --model gpt-5 --threads 8
###############################

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA Test Runner (DA, no search)")
    parser.add_argument("data", type=str, nargs="?", default=DEFAULT_DATA_PATH, help="测试数据文件路径")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="模型名称覆盖默认值")
    parser.add_argument("--threads", type=int, default=MAX_THREADS, help="并行线程数")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="采样温度")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="生成最大 tokens")
    return parser.parse_args()

def simple_match(predicted: str, expected: str) -> bool:
    """简单包含式判断 (大小写不敏感)."""
    predicted = predicted.lower().strip()
    expected = expected.lower().strip()
    return expected in predicted

def call_model(messages: List[Dict[str, Any]], *, model_name: str, max_tokens: int, temperature: float) -> Any:
    """统一模型调用: 支持 search-preview / deep-research 特殊模型名自动分支."""
    retry_count = 0
    delay = RETRY_INITIAL_DELAY
    while True:
        try:
            if "search-preview" in model_name:
                response = client.chat.completions.create(
                    model="gpt-4o-search-preview",
                    web_search_options={},
                    messages=messages,
                )
                return response.choices[0].message
            if "deep-research" in model_name:
                response = client.responses.create(
                    model="o3-deep-research",
                    input=messages,
                    background=True,
                    tools=[{"type": "web_search"}],
                )
                return response.output_text
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message
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
total_count = 0

def process_qa_item(item_data, *, model_name: str, temperature: float, max_tokens: int, total: int):
    idx, item = item_data
    question = item["question"]
    expected_answer = item["answer"]
    if not question.endswith("?"):
        question += "?"
    prompt = (
        "Answer the given question. Provide ONLY the final result inside <answer> and </answer>. "
        "Do not output anything else. For example, <answer> Beijing </answer>. Question: " + question
    )
    print(f"\n=== Question {idx + 1}/{total} ===")
    print(f"Question: {question}")
    print(f"Expected: {expected_answer}")
    messages = [{"role": "user", "content": prompt}]
    response = call_model(messages, model_name=model_name, max_tokens=max_tokens, temperature=temperature)
    search_cnt = 0
    if response:
        if "deep-research" in model_name:
            content = response
        else:
            content = response.content
            if "search-preview" in model_name:
                search_cnt = len(getattr(response, "annotations", []) or [])
        if "<answer>" in content and "</answer>" in content:
            ans = content[content.find("<answer>") + 8: content.find("</answer>")].strip()
        else:
            ans = content.strip()
    else:
        ans = f"Error: Failed to get response from {model_name}"
    is_correct = simple_match(ans, expected_answer)
    print(f"Model Answer: {ans}")
    print(f"Correct: {'✓' if is_correct else '✗'}")
    if "search-preview" in model_name:
        print(f"Search Count: {search_cnt}")
    result = {
        "question": question,
        "expected_answer": expected_answer,
        "model_answer": ans,
        "is_correct": is_correct,
    }
    if "search-preview" in model_name:
        result["search_count"] = search_cnt
    with results_lock:
        results.append(result)
    if is_correct:
        with correct_count_lock:
            global correct_count
            correct_count += 1
    return result

def save_results(*, model_name: str, accuracy: float, total: int, correct: int, data_path: str):
    year_match = re.search(r"(\d{4})", data_path)
    year = year_match.group(1) if year_match else "unknown"
    level_match = re.search(r"(level\d+)", os.path.basename(data_path))
    level = level_match.group(1) if level_match else "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe_name = model_name.replace("-", "_").replace(".", "_").replace("/", "_")
    output_dir = os.path.join("output", year)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{level}_results_DA_{model_safe_name}_{timestamp}.json")
    summary_file = os.path.join(output_dir, f"{level}_summary_DA_{model_safe_name}_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    summary = {
        "model": f"{model_name} (No Search)",
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
    with open(args.data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    global total_count
    total_count = len(test_data["qa_pairs"])
    print(f"Testing {total_count} QA pairs with {model_name} (No Search) using {threads} parallel threads...")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        qa_items = [(idx, item) for idx, item in enumerate(test_data["qa_pairs"])]
        list(
            executor.map(
                lambda data: process_qa_item(
                    data,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    total=total_count,
                ),
                qa_items,
            )
        )
    accuracy = correct_count / total_count if total_count else 0.0
    print(f"\n=== Final Results ({model_name} No Search) ===")
    print(f"Total questions: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    save_results(model_name=model_name, accuracy=accuracy, total=total_count, correct=correct_count, data_path=args.data)


if __name__ == "__main__":
    main()