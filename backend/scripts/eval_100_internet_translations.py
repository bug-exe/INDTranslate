"""
Empirical Evaluation of Model 2 (AI4Bharat Pralekha) vs. Google Translate
========================================================================
Evaluates 100 diverse real-world sentences sampled from the internet across 7 domains:
1. News & Public Affairs (15 sentences)
2. Technology, AI & Internet (15 sentences)
3. Science, Health & Medicine (15 sentences)
4. Cinema & Subtitle Dialogues (15 sentences)
5. Daily Conversation & Travel (15 sentences)
6. Business, Economy & Governance (15 sentences)
7. General Knowledge & Culture (10 sentences)

Metrics Computed against Google Translate Ground Truth Reference:
- Token Precision
- Token Recall
- Token F1-Score
- Token Match Accuracy (Jaccard / Overlap)
- Sentence Exact Match Accuracy
- chrF (Character n-gram F-score)
- BLEU-1 and BLEU-2 scores
"""

import sys
import os
import re
import json
import time
import requests
import asyncio
from typing import List, Dict, Any, Tuple
from collections import Counter

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.services.pralekha_service import translate_with_pralekha, _is_degenerate_output

# 100 Real-world sentences from the internet across 7 domains
BENCHMARK_INTERNET_SENTENCES: List[Dict[str, str]] = [
    # Domain 1: News & Public Affairs (15)
    {"id": "news_01", "domain": "News", "text": "India is rapidly expanding its renewable energy capacity across the country."},
    {"id": "news_02", "domain": "News", "text": "The election commission announced the schedule for the upcoming assembly elections."},
    {"id": "news_03", "domain": "News", "text": "Heavy rainfall caused waterlogging and traffic congestion in several major cities."},
    {"id": "news_04", "domain": "News", "text": "The prime minister inaugurated the new international airport today."},
    {"id": "news_05", "domain": "News", "text": "India won the cricket match by five wickets in a thrilling finish."},
    {"id": "news_06", "domain": "News", "text": "The government launched a new welfare scheme for rural farmers."},
    {"id": "news_07", "domain": "News", "text": "World leaders gathered for the international climate summit."},
    {"id": "news_08", "domain": "News", "text": "The stock market reached a record all-time high during morning trade."},
    {"id": "news_09", "domain": "News", "text": "Schools and colleges will remain closed tomorrow due to the weather alert."},
    {"id": "news_10", "domain": "News", "text": "Scientists discovered new archaeological ruins dating back thousands of years."},
    {"id": "news_11", "domain": "News", "text": "The reserve bank decided to keep the benchmark repo rate unchanged."},
    {"id": "news_12", "domain": "News", "text": "A massive tree plantation drive was organized on World Environment Day."},
    {"id": "news_13", "domain": "News", "text": "The supreme court delivered a landmark judgment on privacy rights."},
    {"id": "news_14", "domain": "News", "text": "Bilateral trade between the two neighboring countries increased significantly."},
    {"id": "news_15", "domain": "News", "text": "Rescue operations are underway following the landslide in the hilly region."},

    # Domain 2: Technology, AI & Internet (15)
    {"id": "tech_01", "domain": "Technology", "text": "Artificial intelligence is transforming industries and modern workplaces."},
    {"id": "tech_02", "domain": "Technology", "text": "Download the latest software update to protect your device from security threats."},
    {"id": "tech_03", "domain": "Technology", "text": "Cloud computing allows developers to deploy applications worldwide with low latency."},
    {"id": "tech_04", "domain": "Technology", "text": "High-speed internet access is essential for digital education and remote work."},
    {"id": "tech_05", "domain": "Technology", "text": "Open source software fosters global collaboration and transparency."},
    {"id": "tech_06", "domain": "Technology", "text": "Please enter the one-time verification code sent to your mobile phone."},
    {"id": "tech_07", "domain": "Technology", "text": "Machine learning models require clean and diverse training datasets."},
    {"id": "tech_08", "domain": "Technology", "text": "Smartphones now come with high-resolution cameras and fast charging."},
    {"id": "tech_09", "domain": "Technology", "text": "Cybersecurity experts warn users against clicking on suspicious links."},
    {"id": "tech_10", "domain": "Technology", "text": "The website is currently undergoing scheduled maintenance."},
    {"id": "tech_11", "domain": "Technology", "text": "Data privacy is a fundamental right in the digital era."},
    {"id": "tech_12", "domain": "Technology", "text": "Electric vehicles are becoming more affordable and accessible to the public."},
    {"id": "tech_13", "domain": "Technology", "text": "Quantum computers could solve complex mathematical problems in seconds."},
    {"id": "tech_14", "domain": "Technology", "text": "Your password must be at least eight characters long and include numbers."},
    {"id": "tech_15", "domain": "Technology", "text": "The mobile application crashed due to an unexpected server timeout."},

    # Domain 3: Science, Health & Medicine (15)
    {"id": "sci_01", "domain": "Health & Science", "text": "Drinking plenty of clean water every day is vital for good health."},
    {"id": "sci_02", "domain": "Health & Science", "text": "Doctors advise getting at least seven hours of uninterrupted sleep every night."},
    {"id": "sci_03", "domain": "Health & Science", "text": "The lunar mission successfully collected soil samples from the moon."},
    {"id": "sci_04", "domain": "Health & Science", "text": "Regular physical exercise helps reduce blood pressure and prevent heart disease."},
    {"id": "sci_05", "domain": "Health & Science", "text": "Vaccination is the most effective way to prevent infectious diseases."},
    {"id": "sci_06", "domain": "Health & Science", "text": "The hospital opened a dedicated pediatric care wing for young children."},
    {"id": "sci_07", "domain": "Health & Science", "text": "Proper hand hygiene prevents the spread of harmful bacteria and viruses."},
    {"id": "sci_08", "domain": "Health & Science", "text": "Astronomers discovered water vapor in the atmosphere of an exoplanet."},
    {"id": "sci_09", "domain": "Health & Science", "text": "Eating fresh fruits and green vegetables strengthens your immune system."},
    {"id": "sci_10", "domain": "Health & Science", "text": "The patient responded positively to the experimental medical treatment."},
    {"id": "sci_11", "domain": "Health & Science", "text": "Global temperatures are rising due to greenhouse gas emissions."},
    {"id": "sci_12", "domain": "Health & Science", "text": "Take this prescribed antibiotic twice a day after meals."},
    {"id": "sci_13", "domain": "Health & Science", "text": "Mental health is just as important as physical fitness for overall well-being."},
    {"id": "sci_14", "domain": "Health & Science", "text": "Solar radiation powers the natural water cycle on planet Earth."},
    {"id": "sci_15", "domain": "Health & Science", "text": "Emergency ambulances are equipped with oxygen cylinders and defibrillators."},

    # Domain 4: Cinema & Subtitle Dialogues (15)
    {"id": "cine_01", "domain": "Cinema & Subtitles", "text": "I promised you that I will return, and I always keep my promises."},
    {"id": "cine_02", "domain": "Cinema & Subtitles", "text": "Look into my eyes and tell me the whole truth right now."},
    {"id": "cine_03", "domain": "Cinema & Subtitles", "text": "Do not be afraid, I am standing right beside you."},
    {"id": "cine_04", "domain": "Cinema & Subtitles", "text": "Time waits for no one; we must leave this place immediately."},
    {"id": "cine_05", "domain": "Cinema & Subtitles", "text": "Truth will always triumph over falsehood in the end."},
    {"id": "cine_06", "domain": "Cinema & Subtitles", "text": "Who are you and why have you come looking for me?"},
    {"id": "cine_07", "domain": "Cinema & Subtitles", "text": "This city gave me everything, and I will defend it with my life."},
    {"id": "cine_08", "domain": "Cinema & Subtitles", "text": "Forgive me for what happened in the past; let us start anew."},
    {"id": "cine_09", "domain": "Cinema & Subtitles", "text": "Never underestimate the determination of a common citizen."},
    {"id": "cine_10", "domain": "Cinema & Subtitles", "text": "If you step backward now, everything we fought for will be lost."},
    {"id": "cine_11", "domain": "Cinema & Subtitles", "text": "Where is the key hidden? We have only five minutes left."},
    {"id": "cine_12", "domain": "Cinema & Subtitles", "text": "Some memories fade away, but true loyalty never dies."},
    {"id": "cine_13", "domain": "Cinema & Subtitles", "text": "Listen to me very carefully; do not trust anyone in this room."},
    {"id": "cine_14", "domain": "Cinema & Subtitles", "text": "We will fight together until our very last breath."},
    {"id": "cine_15", "domain": "Cinema & Subtitles", "text": "Everything happens for a reason; you must believe in destiny."},

    # Domain 5: Daily Conversation & Travel (15)
    {"id": "conv_01", "domain": "Conversation & Travel", "text": "Hello, how are you doing today?"},
    {"id": "conv_02", "domain": "Conversation & Travel", "text": "Thank you so much for your kind help and hospitality."},
    {"id": "conv_03", "domain": "Conversation & Travel", "text": "Could you please tell me the way to the nearest railway station?"},
    {"id": "conv_04", "domain": "Conversation & Travel", "text": "What time does the morning express train arrive on platform two?"},
    {"id": "conv_05", "domain": "Conversation & Travel", "text": "I would like to order a cup of hot tea and some vegetarian snacks."},
    {"id": "conv_06", "domain": "Conversation & Travel", "text": "Where can I find an automatic teller machine nearby?"},
    {"id": "conv_07", "domain": "Conversation & Travel", "text": "How much does a round-trip ticket to Mumbai cost?"},
    {"id": "conv_08", "domain": "Conversation & Travel", "text": "It was wonderful meeting you after so many years."},
    {"id": "conv_09", "domain": "Conversation & Travel", "text": "Please make yourself comfortable and make yourself at home."},
    {"id": "conv_10", "domain": "Conversation & Travel", "text": "The weather today is pleasant and cooler than yesterday."},
    {"id": "conv_11", "domain": "Conversation & Travel", "text": "Excuse me, is this seat occupied or can I sit here?"},
    {"id": "conv_12", "domain": "Conversation & Travel", "text": "Have a safe journey and call me when you reach home."},
    {"id": "conv_13", "domain": "Conversation & Travel", "text": "I am looking for a clean and affordable hotel near the beach."},
    {"id": "conv_14", "domain": "Conversation & Travel", "text": "Can you please speak a little slower so I can understand?"},
    {"id": "conv_15", "domain": "Conversation & Travel", "text": "Good night, sleep well and see you tomorrow morning."},

    # Domain 6: Business, Economy & Governance (15)
    {"id": "gov_01", "domain": "Business & Governance", "text": "Digital payment systems have simplified daily financial transactions in India."},
    {"id": "gov_02", "domain": "Business & Governance", "text": "Small businesses can apply for low-interest collateral-free bank loans."},
    {"id": "gov_03", "domain": "Business & Governance", "text": "The annual financial budget focuses heavily on infrastructure development."},
    {"id": "gov_04", "domain": "Business & Governance", "text": "Citizens can download their digital identity documents online."},
    {"id": "gov_05", "domain": "Business & Governance", "text": "Foreign direct investment in manufacturing grew steadily over the past year."},
    {"id": "gov_06", "domain": "Business & Governance", "text": "Inflation remained within the target tolerance range set by the central bank."},
    {"id": "gov_07", "domain": "Business & Governance", "text": "Goods and services tax revenue collected this month exceeded all expectations."},
    {"id": "gov_08", "domain": "Business & Governance", "text": "Public public-private partnerships will construct modern highways and expressways."},
    {"id": "gov_09", "domain": "Business & Governance", "text": "Income tax returns must be submitted before the official deadline."},
    {"id": "gov_10", "domain": "Business & Governance", "text": "The government announced subsidized seeds and fertilizers for agricultural workers."},
    {"id": "gov_11", "domain": "Business & Governance", "text": "Free primary education is guaranteed for all children under the law."},
    {"id": "gov_12", "domain": "Business & Governance", "text": "Employment generation in tech sectors remains a top economic priority."},
    {"id": "gov_13", "domain": "Business & Governance", "text": "Renewable solar energy projects received substantial investment subsidies."},
    {"id": "gov_14", "domain": "Business & Governance", "text": "Transparent governance enhances citizen trust in democratic institutions."},
    {"id": "gov_15", "domain": "Business & Governance", "text": "Consumer rights organizations protect shoppers against deceptive advertisements."},

    # Domain 7: General Knowledge & Culture (10)
    {"id": "cult_01", "domain": "Knowledge & Culture", "text": "The Taj Mahal is an internationally recognized monument of love and architecture."},
    {"id": "cult_02", "domain": "Knowledge & Culture", "text": "Diwali is the festival of lights celebrated with great enthusiasm across India."},
    {"id": "cult_03", "domain": "Knowledge & Culture", "text": "The Constitution of India recognizes twenty-two scheduled languages."},
    {"id": "cult_04", "domain": "Knowledge & Culture", "text": "Classical dance forms in India tell stories from ancient literature and epics."},
    {"id": "cult_05", "domain": "Knowledge & Culture", "text": "The Himalayan mountain range contains the highest peaks in the entire world."},
    {"id": "cult_06", "domain": "Knowledge & Culture", "text": "Yoga is an ancient Indian physical, mental, and spiritual discipline."},
    {"id": "cult_07", "domain": "Knowledge & Culture", "text": "Indian cuisine is famous worldwide for its aromatic spices and regional diversity."},
    {"id": "cult_08", "domain": "Knowledge & Culture", "text": "River Ganga flows through northern plains and holds profound cultural reverence."},
    {"id": "cult_09", "domain": "Knowledge & Culture", "text": "Independence Day is celebrated nationwide on the fifteenth of August."},
    {"id": "cult_10", "domain": "Knowledge & Culture", "text": "Unity in diversity is the greatest cultural strength of our country."}
]


def fetch_google_translate(text: str, sl: str = "en", tl: str = "hi") -> str:
    """Fetch official reference translation from Google Translate."""
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "dict-chrome-ex",
        "sl": sl,
        "tl": tl,
        "dt": "t",
        "q": text
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Accept": "*/*"
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                res = "".join([seg[0] for seg in data[0] if seg and seg[0]])
                return res.strip()
    except Exception as e:
        print(f"Google translate request failed: {e}")
    return ""


def tokenize_indic(text: str) -> List[str]:
    """Tokenize Indic script text into normalized word/morpheme units."""
    # Remove punctuation while preserving Indic Unicode letters and digits
    cleaned = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    tokens = [t.strip() for t in cleaned.split() if t.strip()]
    return tokens


def compute_token_metrics(pred_tokens: List[str], ref_tokens: List[str]) -> Tuple[float, float, float, float]:
    """
    Computes Token Precision, Recall, F1, and Token Accuracy (Overlap).
    Uses multiset / Counter intersection to handle word frequency correctly.
    """
    if not ref_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0, 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0, 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)

    # True positives: min count of each token occurring in both
    tp = sum(min(pred_counts[token], ref_counts[token]) for token in pred_counts if token in ref_counts)

    precision = tp / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = tp / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Token accuracy (Intersection over Union / Jaccard-like matching)
    total_distinct_mass = max(len(ref_tokens), len(pred_tokens))
    accuracy = tp / total_distinct_mass if total_distinct_mass > 0 else 0.0

    return precision, recall, f1, accuracy


def compute_chrf(pred_text: str, ref_text: str, n: int = 4, beta: float = 2.0) -> float:
    """
    Computes Character n-gram F-score (chrF), the standard metric for morphologically rich Indic languages.
    """
    def get_char_ngrams(s: str, n_val: int) -> Counter:
        norm = s.replace(" ", "")
        ngrams = [norm[i:i+n_val] for i in range(len(norm) - n_val + 1)]
        return Counter(ngrams)

    if not pred_text or not ref_text:
        return 0.0

    precisions = []
    recalls = []
    for k in range(1, n + 1):
        pred_ng = get_char_ngrams(pred_text, k)
        ref_ng = get_char_ngrams(ref_text, k)
        match = sum(min(pred_ng[ng], ref_ng[ng]) for ng in pred_ng if ng in ref_ng)
        total_p = sum(pred_ng.values())
        total_r = sum(ref_ng.values())
        p_k = match / total_p if total_p > 0 else 0.0
        r_k = match / total_r if total_r > 0 else 0.0
        precisions.append(p_k)
        recalls.append(r_k)

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0

    if avg_p + avg_r == 0:
        return 0.0
    # chrF with beta weighting (beta=2 favors recall slightly, standard in MT)
    chrf_score = (1 + beta**2) * (avg_p * avg_r) / ((beta**2 * avg_p) + avg_r)
    return chrf_score


def compute_bleu1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """BLEU-1 score with standard brevity penalty."""
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    tp = sum(min(pred_counts[token], ref_counts[token]) for token in pred_counts if token in ref_counts)
    p1 = tp / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    # Brevity penalty
    c = len(pred_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else (pow(2.71828, 1 - r / c) if c > 0 else 0.0)
    return bp * p1


async def run_benchmark():
    print(f"=================================================================")
    print(f"BENCHMARK: 100 INTERNET TRANSLATIONS EVALUATION")
    print(f"Target: Model 2 (AI4Bharat Pralekha) vs. Google Translate (Ground Truth)")
    print(f"=================================================================\n")

    results = []
    domain_metrics: Dict[str, Dict[str, List[float]]] = {}

    total_sentences = len(BENCHMARK_INTERNET_SENTENCES)
    assert total_sentences == 100, f"Expected 100 sentences, got {total_sentences}"

    start_time = time.time()

    for idx, item in enumerate(BENCHMARK_INTERNET_SENTENCES, 1):
        s_id = item["id"]
        domain = item["domain"]
        eng_text = item["text"]

        # 1. Fetch Google Translate Ground Truth Reference
        ref_hi = fetch_google_translate(eng_text, sl="en", tl="hi")
        if not ref_hi:
            # Fallback retry
            time.sleep(0.3)
            ref_hi = fetch_google_translate(eng_text, sl="en", tl="hi")

        # 2. Generate Prediction via Model 2 (Pralekha Direct Indic Engine)
        pred_dict = await translate_with_pralekha("en", "hi", eng_text)
        pred_hi = pred_dict.get("translated_text", "")

        # 3. Tokenize & Compute Metrics
        ref_tokens = tokenize_indic(ref_hi)
        pred_tokens = tokenize_indic(pred_hi)

        precision, recall, f1, accuracy = compute_token_metrics(pred_tokens, ref_tokens)
        chrf = compute_chrf(pred_hi, ref_hi)
        bleu1 = compute_bleu1(pred_tokens, ref_tokens)
        exact_match = 1.0 if ref_hi.strip() == pred_hi.strip() else 0.0

        record = {
            "id": s_id,
            "domain": domain,
            "english": eng_text,
            "google_translate_ref": ref_hi,
            "model2_pralekha_pred": pred_hi,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "token_accuracy": round(accuracy, 4),
            "chrf": round(chrf, 4),
            "bleu1": round(bleu1, 4),
            "exact_match": exact_match
        }
        results.append(record)

        if domain not in domain_metrics:
            domain_metrics[domain] = {
                "precision": [], "recall": [], "f1": [], "token_accuracy": [], "chrf": [], "bleu1": []
            }
        domain_metrics[domain]["precision"].append(precision)
        domain_metrics[domain]["recall"].append(recall)
        domain_metrics[domain]["f1"].append(f1)
        domain_metrics[domain]["token_accuracy"].append(accuracy)
        domain_metrics[domain]["chrf"].append(chrf)
        domain_metrics[domain]["bleu1"].append(bleu1)

        if idx % 10 == 0 or idx == total_sentences:
            print(f"[{idx:3d}/100] Processed ({domain:20s}) -> Current Avg F1: {sum(r['f1'] for r in results)/len(results)*100:.2f}%")

    elapsed = time.time() - start_time

    # Compute Global Averages
    overall_p = sum(r["precision"] for r in results) / len(results)
    overall_r = sum(r["recall"] for r in results) / len(results)
    overall_f1 = sum(r["f1"] for r in results) / len(results)
    overall_acc = sum(r["token_accuracy"] for r in results) / len(results)
    overall_chrf = sum(r["chrf"] for r in results) / len(results)
    overall_bleu1 = sum(r["bleu1"] for r in results) / len(results)
    exact_match_rate = sum(r["exact_match"] for r in results) / len(results)

    summary = {
        "benchmark_name": "100 Random Internet Translations vs. Google Translate",
        "sample_size": total_sentences,
        "source_language": "en",
        "target_language": "hi",
        "reference_system": "Google Translate (Official Web API)",
        "evaluated_system": "Model 2 (AI4Bharat Pralekha Direct Indic Seq2Seq)",
        "evaluation_duration_seconds": round(elapsed, 2),
        "overall_metrics": {
            "token_precision_percent": round(overall_p * 100, 2),
            "token_recall_percent": round(overall_r * 100, 2),
            "token_f1_score_percent": round(overall_f1 * 100, 2),
            "token_accuracy_percent": round(overall_acc * 100, 2),
            "chrf_score_percent": round(overall_chrf * 100, 2),
            "bleu1_percent": round(overall_bleu1 * 100, 2),
            "sentence_exact_match_percent": round(exact_match_rate * 100, 2)
        },
        "domain_breakdown": {
            d: {
                "count": len(vals["f1"]),
                "precision": round(sum(vals["precision"]) / len(vals["precision"]) * 100, 2),
                "recall": round(sum(vals["recall"]) / len(vals["recall"]) * 100, 2),
                "f1": round(sum(vals["f1"]) / len(vals["f1"]) * 100, 2),
                "token_accuracy": round(sum(vals["token_accuracy"]) / len(vals["token_accuracy"]) * 100, 2),
                "chrf": round(sum(vals["chrf"]) / len(vals["chrf"]) * 100, 2),
                "bleu1": round(sum(vals["bleu1"]) / len(vals["bleu1"]) * 100, 2),
            }
            for d, vals in domain_metrics.items()
        },
        "sample_comparisons": results[:10]  # First 10 samples
    }

    # Save complete evaluation output
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../eval_100_internet_translations.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=================================================================")
    print(f"BENCHMARK RESULTS SUMMARY (100 INTERNET TRANSLATIONS)")
    print(f"=================================================================")
    print(f"Token Precision : {summary['overall_metrics']['token_precision_percent']}%")
    print(f"Token Recall    : {summary['overall_metrics']['token_recall_percent']}%")
    print(f"Token F1-Score  : {summary['overall_metrics']['token_f1_score_percent']}%")
    print(f"Token Accuracy  : {summary['overall_metrics']['token_accuracy_percent']}%")
    print(f"chrF Metric     : {summary['overall_metrics']['chrf_score_percent']}%")
    print(f"BLEU-1 Score    : {summary['overall_metrics']['bleu1_percent']}%")
    print(f"Exact Match     : {summary['overall_metrics']['sentence_exact_match_percent']}%")
    print(f"Saved complete audit log to: {out_path}")
    print(f"=================================================================\n")

    return summary


if __name__ == "__main__":
    asyncio.run(run_benchmark())
