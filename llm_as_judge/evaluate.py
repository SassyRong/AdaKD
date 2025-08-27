import argparse
import json
import re
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm
import httpx
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI
import asyncio


JUDGE_SYSTEM_PROMPT_WITH_GT = """You are an impartial judge. Your task is to evaluate two AI assistants' responses.

First, provide a step-by-step thinking process within <think></think> tags. **Your thinking process MUST be concise and under 120 words.**
During your thinking process, you MUST follow these rules:
- **Avoid Position Bias**: The order of the responses (A vs. B) must NOT influence your decision.
- **Ignore Length Bias**: Do NOT favor longer or shorter responses. Focus on content quality.
- **Strive to make a choice**: If there is any meaningful difference in quality, even a subtle one, you should choose a winner.
- Compare each response to the ground truth and the user's question.

After your thinking, on a new line, provide your final verdict. The verdict MUST be one of "[[A]]", "[[B]]", or "[[C]]" and nothing else.
- **[[A]]**: Assistant A is better.
- **[[B]]**: Assistant B is better.
- **[[C]]**: A tie. Use this ONLY if the responses are of truly indistinguishable quality or are equally and completely incorrect.

Example:
<think>
The user asks for the start date of Virgin Australia. The ground truth confirms August 31, 2000. Both Assistant A and B provide this correct date. The only difference is a period, which is a minor stylistic choice. Length and position are irrelevant here as the core information is identical and correct in both. This results in a tie.
</think>
[[C]]
"""


def load_data_with_gt(
    questions_file: str, 
    answers_file1: str, 
    answers_file2: str
) -> List[Tuple[str, str, str, str]]:

    questions = []
    ground_truths = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            if 'input' in data and data['input']:
                full_question = f"{data['instruction']}\n\nInput:\n{data['input']}"
            else:
                full_question = data['instruction']
            
            questions.append(full_question)

            ground_truths.append(data['output'])

    answers1 = []
    with open(answers_file1, 'r', encoding='utf-8') as f:
        for line in f:
            answers1.append(json.loads(line)['text'])

    answers2 = []
    with open(answers_file2, 'r', encoding='utf-8') as f:
        for line in f:
            answers2.append(json.loads(line)['text'])
    
    if not (len(questions) == len(ground_truths) == len(answers1) == len(answers2)):
        raise ValueError(f"Error: The number of lines in files do not match. "
                         f"Questions: {len(questions)}, Answers A: {len(answers1)}, Answers B: {len(answers2)}")
        
    return list(zip(questions, ground_truths, answers1, answers2))

def parse_judge_output(text: str) -> Tuple[str, str]:
    explanation = text.strip()
    think_end_tag = '</think>'
    tag_pos = text.rfind(think_end_tag)
    search_area = text
    if tag_pos != -1:
        search_area = text[tag_pos + len(think_end_tag):]
    match = re.search(r'\[\[([ABC])\]\]', search_area)
    if match:
        verdict = match.group(0)
    else:
        fallback_match = re.search(r'\[\[([ABC])\]\]', text)
        verdict = fallback_match.group(0) if fallback_match else "[[Parsing Failed]]"
    return explanation, verdict

def create_qwen_judge_prompt(question, ground_truth, answer_a, answer_b) -> str:
    user_content = f"""[User Question]\n{question}\n\n[Ground Truth Answer]\n{ground_truth}\n\n[The Start of Assistant A’s Answer]\n{answer_a}\n[The End of Assistant A’s Answer]\n\n[The Start of Assistant B’s Answer]\n{answer_b}\n[The End of Assistant B’s Answer]"""
    return (
        f"<|im_start|>system\n{JUDGE_SYSTEM_PROMPT_WITH_GT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
async def main():
    parser = argparse.ArgumentParser(description="Debiased model evaluation using dual-pass (swap) method.")
    parser.add_argument('--questions-file', type=str, required=True, help="Path to questions with ground truth answers in JSONL format.")
    parser.add_argument('--answers-file-a', type=str, required=True, help="Path to answers from Model A.")
    parser.add_argument('--answers-file-b', type=str, required=True, help="Path to answers from Model B.")
    parser.add_argument('--api-base-url', type=str, default="http://127.0.0.1:8009/v1")
    parser.add_argument('--api-key', type=str, default="not-needed")
    parser.add_argument('--served-model-name', type=str, default="Qwen3-32B")
    parser.add_argument('--output-file', type=str, default="evaluation_results.jsonl")
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--concurrency', type=int, default=16)
    
    args = parser.parse_args()

    custom_async_http_client = httpx.AsyncClient(trust_env=False)
    aclient = AsyncOpenAI(base_url=args.api_base_url, api_key=args.api_key, http_client=custom_async_http_client)

    evaluation_data = load_data_with_gt(args.questions_file, args.answers_file_a, args.answers_file_b)
    
    tasks = []
    print("Preparing tasks for dual-pass evaluation (normal and swapped)...")
    for i, (question, ground_truth, answer_a, answer_b) in enumerate(evaluation_data):
        prompt_normal = create_qwen_judge_prompt(question, ground_truth, answer_a, answer_b)
        tasks.append({'id': i, 'type': 'normal', 'prompt': prompt_normal})
        
        prompt_swapped = create_qwen_judge_prompt(question, ground_truth, answer_b, answer_a)
        tasks.append({'id': i, 'type': 'swapped', 'prompt': prompt_swapped})

    semaphore = asyncio.Semaphore(args.concurrency)
    
    async def get_evaluation(task):
        async with semaphore:
            try:
                response = await aclient.completions.create(
                    model=args.served_model_name,
                    prompt=task['prompt'],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                return task['id'], task['type'], response.choices[0].text
            except Exception as e:
                return task['id'], task['type'], f"API_CALL_FAILED: {e}"

    print(f"Sending {len(tasks)} total requests (2 per question) with concurrency {args.concurrency}...")
    all_raw_responses = await async_tqdm.gather(*[get_evaluation(task) for task in tasks])

    results_by_id = {}
    for i, data in enumerate(evaluation_data):
        results_by_id[i] = {
            "question": data[0],
            "ground_truth": data[1],
            "answer_model_A": data[2],
            "answer_model_B": data[3], 
        }

    for task_id, task_type, response_text in all_raw_responses:
        if task_type == 'normal':
            results_by_id[task_id]['normal_response'] = response_text
        else: # swapped
            results_by_id[task_id]['swapped_response'] = response_text

    print("All requests completed. Adjudicating final results...")
    
    final_win_counts = {"Model A": 0, "Model B": 0, "Tie": 0}
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for i in range(len(evaluation_data)):
            result_item = results_by_id[i]
            
            normal_exp, normal_verdict = parse_judge_output(result_item.get('normal_response', ''))
            swapped_exp, swapped_verdict = parse_judge_output(result_item.get('swapped_response', ''))

            debiased_verdict = "Tie" 
            if normal_verdict == '[[A]]' and swapped_verdict == '[[B]]':
                debiased_verdict = "Model A Wins"
                final_win_counts["Model A"] += 1
            elif normal_verdict == '[[B]]' and swapped_verdict == '[[A]]':
                debiased_verdict = "Model B Wins"
                final_win_counts["Model B"] += 1
            else:
                final_win_counts["Tie"] += 1

            final_result_obj = {
                "id": i,
                "question": result_item["question"],
                "ground_truth": result_item["ground_truth"],
                "answer_model_A": result_item["answer_model_A"],
                "answer_model_B": result_item["answer_model_B"],
                "normal_run_verdict": normal_verdict,
                "swapped_run_verdict": swapped_verdict,
                "debiased_verdict": debiased_verdict,
                "normal_run_explanation": normal_exp,
                "swapped_run_explanation": swapped_exp,
            }
            f_out.write(json.dumps(final_result_obj, ensure_ascii=False) + '\n')
            f_out.flush()

    print("\n--- Debiased Evaluation Summary ---")
    print(f"Model A Wins: {final_win_counts['Model A']}")
    print(f"Model B Wins: {final_win_counts['Model B']}")
    print(f"Ties (including bias/failures): {final_win_counts['Tie']}")
    print("-----------------------------------")
    print(f"Detailed debiased results saved to {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())