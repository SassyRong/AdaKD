import string
import json
import os
import argparse
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from transformers import AutoTokenizer
from transformers import logging
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def rouge(prediction, ground_truth, xlingual=False):
    scorer = default_rouge_scorer
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure

def bleu(prediction, ground_truth, xlingual=False):
    prediction = word_tokenize(prediction)
    ground_truth = word_tokenize(ground_truth)
    cc = SmoothingFunction()
    return sentence_bleu(ground_truth, prediction, smoothing_function=cc.method1)

def meteor(prediction, ground_truth, xlingual=False):
    prediction = word_tokenize(prediction)
    ground_truth = word_tokenize(ground_truth)
    return meteor_score([ground_truth], prediction)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics_rouge(predictions, references, xlingual=False):
    # assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    
    min_length = min(len((predictions)), len(references))
    predictions = predictions[:min_length]
    references = references[:min_length]
    
    em, rougeL = 0, 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        em += metric_max_over_ground_truths(
            exact_match, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rouge, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    em = 100.0 * em / len(references)
    rougeL = 100.0 * rougeL / len(references)
    metrics = {"exact_match": em, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics

def compute_metrics_bert(predictions, references, xlingual=False):
    min_length = min(len(predictions), len(references))
    predictions = predictions[:min_length]
    references = references[:min_length]

    logging.set_verbosity_error()
    model_type = "bert-base-multilingual-cased" if xlingual else "roberta-large"
    lang = "en" if not xlingual else "zh"
    scorer = BERTScorer(model_type, lang=lang)
    _, _, bert_scores = scorer.score(predictions, references)
    bert_scores = 100.0 * bert_scores.mean().item()
    return {"bert_score": round(bert_scores, 4)}

def compute_metrics_bleu(predictions, references, xlingual=False):
    min_length = min(len(predictions), len(references))
    predictions = predictions[:min_length]
    references = references[:min_length]

    smoothie = SmoothingFunction().method1
    refs_tokenized = [[word_tokenize(ref) for ref in refs] for refs in references]
    preds_tokenized = [word_tokenize(pred) for pred in predictions]
    bleu = corpus_bleu(
        refs_tokenized, preds_tokenized,
        smoothing_function=smoothie,
        weights=(0.25, 0.25, 0.25, 0.25)
    ) * 100
    return {"bleu": round(bleu, 4)}
    
def compute_metrics_meteor(predictions, references, xlingual=False):
    min_length = min(len(predictions), len(references))
    predictions = predictions[:min_length]
    references = references[:min_length]

    mts = 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        mts += metric_max_over_ground_truths(meteor, pred, gold, xlingual=xlingual)
    mts = 100.0 * mts / len(references)
    return {"meteor": round(mts, 4)}

    

def compute_grouped_metrics(predictions, references, groups, xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics_rouge(task_predictions, task_references, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_file", required=True,
        help="Jsonl file with each line corresponding to a prediction. " 
             "Each json object should have an `id` and a `prediction` key.")
    parser.add_argument(
        "--reference_file", required=True,
        help="Jsonl file with each line corresponding to a reference. " 
             "Each json object should have an `id` and a `references` key. "
             "`task_id`, `task_category` and `task_track` are optional, which will be used to "
             "compute the per-task performance, per-category performance and the performance for default (english) / xlingual Tracks.")
    parser.add_argument(
        "--output_file",
        help="Jsonl file to write the results to.")
    parser.add_argument(
        "--model_name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    references = []
    with open(args.reference_file) as fin:
        for line in fin:
            instance = json.loads(line)
            if isinstance(instance["output"], list):
                references.append(instance["output"])
            else:
                references.append([instance["output"]])

    predictions = []
    with open(args.prediction_file) as fin:
        for line in fin:
            prediction = json.loads(line)
            predictions.append(prediction["text"])

    predictions = predictions[:1000]

    references = references[:len(predictions)]

    results = compute_metrics_rouge(predictions, references, xlingual=False)

    print(results)

    if args.output_file:
        os.makedirs(args.output_file, exist_ok=True)
        with open(os.path.join(args.output_file, f"{args.model_name}.json"), "w") as fout:
            json.dump(results, fout, indent=2)
            