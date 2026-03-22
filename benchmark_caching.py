"""Benchmark: Context caching cost savings for agent_core.

Simulates multi-round agent conversations with caching ON vs OFF,
captures real token usage from Vertex AI usage_metadata, and
computes dollar savings.

Usage:
    python benchmark_caching.py

Requires GOOGLE_PROJECT_ID in environment or .env file.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))
ldia_env = Path(__file__).parent.parent / "ldia" / ".env"
if ldia_env.exists():
    load_dotenv(ldia_env)

from google import genai
from google.genai import types

# --- Configuration ---

PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID", "exp001-429822")
LOCATION = os.environ.get("GOOGLE_LOCATION", "global")
MODEL = "gemini-3-pro-preview"

# Pricing per million tokens (gemini-3-pro-preview, prompts <= 200K)
PRICE_INPUT_STANDARD = 2.00
PRICE_INPUT_CACHED = 0.20
PRICE_OUTPUT = 12.00
PRICE_CACHE_STORAGE = 4.50

# Large system prompt to simulate a real agent with context
SYSTEM_PROMPT = """You are a senior data analyst assistant for ML researchers.
You help analyze large local datasets (TB-scale) without cloud uploads.

Your capabilities:
- Explore directory structures and file manifests
- Profile CSV, text, image, and graph datasets
- Generate and execute Python code in a sandboxed environment
- Create statistical summaries and visualizations
- Track artifacts (manifests, summaries, reports) across sessions

Guidelines:
- Always start by understanding the data structure before analysis
- Use memory-efficient approaches (chunked reading, sampling) for large files
- Prefer pandas for tabular data, PIL for images, networkx for graphs
- Generate visualizations using matplotlib/seaborn
- Store intermediate results as artifacts for later reference
- When writing code, include error handling for edge cases
- Explain your reasoning before executing analysis steps

Available tools:
- scan_directory: List files and compute basic statistics
- profile_csv: Generate statistical profile of a CSV file
- execute_code: Run Python code in sandbox (30s timeout)
- save_artifact: Persist analysis results
- get_artifact: Retrieve previous artifacts
- list_artifacts: Show all stored artifacts
- run_comparison: Compare two datasets using sweetviz

Current project context:
- Working directory: /data/wine_quality/
- Dataset: Wine Quality (UCI ML Repository)
- Files: winequality-red.csv (1599 rows, 12 cols), winequality-white.csv (4898 rows, 12 cols)
- Goal: Comprehensive EDA and quality prediction feature analysis
- Previous artifacts: data_manifest (v1), red_wine_profile (v1)

Additional context from previous analysis sessions:
The wine quality dataset contains physicochemical properties of Portuguese "Vinho Verde" wine.
Input variables are based on physicochemical tests:
1. fixed acidity (tartaric acid - g/dm³)
2. volatile acidity (acetic acid - g/dm³)
3. citric acid (g/dm³)
4. residual sugar (g/dm³)
5. chlorides (sodium chloride - g/dm³)
6. free sulfur dioxide (mg/dm³)
7. total sulfur dioxide (mg/dm³)
8. density (g/cm³)
9. pH
10. sulphates (potassium sulphate - g/dm³)
11. alcohol (% by volume)
Output variable: quality (score between 0 and 10 based on sensory data)

Previous findings from red wine analysis:
- Quality distribution is concentrated around 5-6 (mean 5.64)
- Alcohol has strongest positive correlation with quality (0.476)
- Volatile acidity has strongest negative correlation (-0.391)
- No missing values across all columns
- Outliers detected in residual sugar and chlorides
- Fixed acidity ranges from 4.6 to 15.9 with mean 8.32
- pH ranges from 2.74 to 4.01 with mean 3.31
- Density is tightly distributed around 0.9967

Red wine statistical summary:
  fixed_acidity:    mean=8.32, std=1.74, min=4.6, max=15.9
  volatile_acidity: mean=0.528, std=0.179, min=0.12, max=1.58
  citric_acid:      mean=0.271, std=0.195, min=0.0, max=1.0
  residual_sugar:   mean=2.539, std=1.41, min=0.9, max=15.5
  chlorides:        mean=0.087, std=0.047, min=0.012, max=0.611
  free_sulfur_dioxide: mean=15.87, std=10.46, min=1.0, max=72.0
  total_sulfur_dioxide: mean=46.47, std=32.9, min=6.0, max=289.0
  density:          mean=0.9967, std=0.0019, min=0.9901, max=1.0037
  pH:               mean=3.311, std=0.154, min=2.74, max=4.01
  sulphates:        mean=0.658, std=0.17, min=0.33, max=2.0
  alcohol:          mean=10.42, std=1.07, min=8.4, max=14.9
  quality:          mean=5.636, std=0.808, min=3, max=8

White wine statistical summary:
  fixed_acidity:    mean=6.85, std=0.84, min=3.8, max=14.2
  volatile_acidity: mean=0.278, std=0.101, min=0.08, max=1.1
  citric_acid:      mean=0.334, std=0.121, min=0.0, max=1.66
  residual_sugar:   mean=6.391, std=5.072, min=0.6, max=65.8
  chlorides:        mean=0.046, std=0.022, min=0.009, max=0.346
  free_sulfur_dioxide: mean=35.31, std=17.01, min=2.0, max=289.0
  total_sulfur_dioxide: mean=138.36, std=42.5, min=9.0, max=440.0
  density:          mean=0.994, std=0.003, min=0.987, max=1.039
  pH:               mean=3.188, std=0.151, min=2.72, max=3.82
  sulphates:        mean=0.490, std=0.115, min=0.22, max=1.08
  alcohol:          mean=10.51, std=1.23, min=8.0, max=14.2
  quality:          mean=5.878, std=0.886, min=3, max=9

Correlation analysis results (merged dataset):
  alcohol vs quality:           0.444
  density vs quality:          -0.307
  volatile_acidity vs quality: -0.265
  chlorides vs quality:        -0.201
  citric_acid vs quality:       0.086
  total_sulfur_dioxide:        -0.175
  fixed_acidity:               -0.077
  residual_sugar:              -0.037
  free_sulfur_dioxide:          0.055
  pH:                           0.019
  sulphates:                    0.038

Model performance benchmarks from prior sessions:
  Logistic Regression: accuracy=0.762, f1=0.741
  Random Forest: accuracy=0.847, f1=0.835, feature_importance=[alcohol:0.189, sulphates:0.112, volatile_acidity:0.108]
  Gradient Boosting: accuracy=0.861, f1=0.853
  Cross-validation (5-fold, GB): 0.853 ± 0.012
"""

# Conversations: each prompt is the user turn, model responds naturally
CONVERSATIONS = {
    "short_3_rounds": [
        "What are the main differences between the red and white wine datasets in terms of their chemical properties?",
        "Based on what you know, which features should I focus on for building a quality prediction model? Explain your reasoning with the correlation data.",
        "Write me a summary report of all findings so far, including the model benchmarks. Format it as a structured document with sections.",
    ],
    "medium_7_rounds": [
        "Give me an overview of the wine quality datasets we're working with.",
        "What does the statistical summary tell us about red wine acidity levels? Are there any concerns?",
        "Compare the sulfur dioxide levels between red and white wines. Why might they differ?",
        "Looking at the correlation results, what's surprising? Are there features that correlate differently than expected?",
        "If I wanted to engineer new features from the existing ones, what combinations would you suggest based on the analysis?",
        "Walk me through the model comparison results. Why does gradient boosting outperform random forest here?",
        "Create a comprehensive action plan for improving the model performance beyond 86%. What should we try next?",
    ],
    "long_12_rounds": [
        "Let's start from the beginning. Describe the wine quality dataset — what is it, where does it come from, and what are we trying to predict?",
        "Walk me through the red wine statistics column by column. What stands out?",
        "Now do the same for white wine. How does it compare to red?",
        "The quality scores seem concentrated around 5-6. Is this a class imbalance problem? How should we handle it?",
        "Explain the correlation patterns. Why does alcohol positively correlate with quality but density negatively correlate?",
        "Are there any multicollinearity concerns in this dataset? Which feature pairs are highly correlated with each other?",
        "Let's talk about outliers. Based on the stats, which features have the most extreme outliers and how should we handle them?",
        "What preprocessing pipeline would you recommend before modeling? Consider scaling, encoding, outlier handling, and feature selection.",
        "Compare logistic regression vs random forest vs gradient boosting. What are the tradeoffs for this specific dataset?",
        "The gradient boosting model gets 86.1% accuracy. What are the most likely sources of the remaining 13.9% error?",
        "If we had more data or different features, what additional variables might improve quality prediction?",
        "Write a final executive summary of this entire analysis project, including methodology, key findings, model results, and next steps.",
    ],
    "extended_15_rounds": [
        "Start with a high-level overview of the wine quality prediction project.",
        "Describe the physicochemical properties measured in the dataset and their typical ranges.",
        "What's the quality distribution like? Show me the breakdown for both red and white wines.",
        "Explain the key differences between red and white wine in terms of acidity and sulfur dioxide.",
        "Walk me through the correlation analysis. What are the top 5 positive and negative correlators?",
        "How does residual sugar differ between red and white wines? What's the biological explanation?",
        "Let's discuss density. Why is it so tightly distributed and why does it negatively correlate with quality?",
        "Are there interaction effects between features? For example, does the relationship between alcohol and quality change at different pH levels?",
        "Propose a feature engineering strategy. What derived features could capture domain knowledge?",
        "Evaluate the three models we've tested. What are the strengths and weaknesses of each?",
        "The random forest feature importances show alcohol at 0.189. Is this relative importance or absolute? How should we interpret it?",
        "What regularization techniques might help with gradient boosting? Would early stopping or learning rate tuning make a difference?",
        "If we treated quality as a regression problem instead of classification, how would our approach change?",
        "Compare our results to published benchmarks for the UCI wine quality dataset. Are we competitive?",
        "Draft a complete technical report: introduction, data description, EDA findings, preprocessing, modeling, results, and future work.",
    ],
}


def run_conversation(client, prompts, system_prompt, use_cache=False):
    """Run a multi-round conversation and collect token usage stats.

    Returns list of per-round stats and (optionally) the final cache name to clean up.
    """
    history = []
    stats = []
    active_cache_name = None
    cache_prefix_len = 0

    for i, prompt in enumerate(prompts):
        user_content = types.Content(
            role="user", parts=[types.Part.from_text(text=prompt)]
        )
        history.append(user_content)

        # Build config and contents
        if use_cache and active_cache_name:
            config = types.GenerateContentConfig(
                cached_content=active_cache_name,
                temperature=0.7,
                max_output_tokens=4096,
            )
            send_contents = history[cache_prefix_len:]
        else:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=4096,
            )
            send_contents = history

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=send_contents,
                config=config,
            )

            meta = response.usage_metadata
            prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
            completion_tokens = getattr(meta, "candidates_token_count", 0) or 0
            cached_tokens = getattr(meta, "cached_content_token_count", 0) or 0

            # Add model response to history
            if response.candidates and response.candidates[0].content:
                history.append(response.candidates[0].content)
            else:
                # Model returned empty content — add placeholder to maintain alternation
                history.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=response.text or "I understand.")],
                ))

            stats.append({
                "round": i + 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "used_cache": use_cache and active_cache_name is not None,
            })

            cache_label = f" (cached: {cached_tokens:,})" if cached_tokens else ""
            mode = "cached" if use_cache else "no cache"
            print(f"  Round {i+1:>2} ({mode}): {prompt_tokens:>8,} in{cache_label:20s} / {completion_tokens:>6,} out")

        except Exception as e:
            print(f"  Round {i+1:>2} ERROR: {e}")
            # Add placeholder to maintain alternation
            history.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text="Error occurred.")],
            ))
            stats.append({
                "round": i + 1,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "used_cache": False,
            })

        # Post-round caching (only in cached mode)
        if use_cache:
            try:
                token_count = client.models.count_tokens(
                    model=MODEL, contents=history
                ).total_tokens

                if token_count >= 32_768:
                    # Delete old cache
                    if active_cache_name:
                        try:
                            client.caches.delete(name=active_cache_name)
                        except Exception:
                            pass

                    cache = client.caches.create(
                        model=MODEL,
                        config=types.CreateCachedContentConfig(
                            contents=list(history),
                            system_instruction=system_prompt,
                            ttl="300s",
                        ),
                    )
                    active_cache_name = cache.name
                    cache_prefix_len = len(history)
                    print(f"           Cache created: {token_count:,} tokens cached")
            except Exception as e:
                print(f"           Cache creation failed: {e}")

        time.sleep(0.5)

    return stats, active_cache_name


def compute_costs(stats):
    """Compute dollar costs from token usage stats."""
    total_input = sum(s["prompt_tokens"] for s in stats)
    total_output = sum(s["completion_tokens"] for s in stats)
    total_cached = sum(s["cached_tokens"] for s in stats)
    standard_input = total_input - total_cached

    cost_input_standard = (standard_input / 1_000_000) * PRICE_INPUT_STANDARD
    cost_input_cached = (total_cached / 1_000_000) * PRICE_INPUT_CACHED
    cost_output = (total_output / 1_000_000) * PRICE_OUTPUT

    # Estimate storage cost (cache lives ~5 min per round average)
    num_rounds = len(stats)
    avg_cache_tokens = total_cached / max(num_rounds, 1)
    cache_hours = (num_rounds * 5) / 60
    cost_storage = (avg_cache_tokens / 1_000_000) * PRICE_CACHE_STORAGE * cache_hours

    return {
        "total_input": total_input,
        "total_output": total_output,
        "total_cached": total_cached,
        "standard_input": standard_input,
        "cost_input_standard": cost_input_standard,
        "cost_input_cached": cost_input_cached,
        "cost_output": cost_output,
        "cost_storage": cost_storage,
        "cost_total": cost_input_standard + cost_input_cached + cost_output + cost_storage,
    }


def run_benchmark():
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    print("=" * 80)
    print("CONTEXT CACHING BENCHMARK — agent_core")
    print(f"Model: {MODEL}")
    print(f"Project: {PROJECT_ID}")
    print(f"Cache threshold: 32,768 tokens")
    print(f"Pricing: ${PRICE_INPUT_STANDARD}/M input, ${PRICE_INPUT_CACHED}/M cached, ${PRICE_OUTPUT}/M output")
    print("=" * 80)

    # Verify API access
    try:
        resp = client.models.count_tokens(
            model=MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text="test")])],
        )
        print(f"API connected. System prompt tokens: ~{len(SYSTEM_PROMPT) // 4}")
    except Exception as e:
        print(f"ERROR: Cannot connect to Vertex AI: {e}")
        sys.exit(1)

    all_results = {}

    for scenario_name, prompts in CONVERSATIONS.items():
        print(f"\n{'─' * 80}")
        print(f"Scenario: {scenario_name} ({len(prompts)} rounds)")
        print(f"{'─' * 80}")

        # Run WITHOUT caching
        print(f"\n  --- Without caching ---")
        no_cache_stats, _ = run_conversation(client, prompts, SYSTEM_PROMPT, use_cache=False)

        # Run WITH caching
        print(f"\n  --- With caching ---")
        cached_stats, cache_name = run_conversation(client, prompts, SYSTEM_PROMPT, use_cache=True)

        # Clean up remaining cache
        if cache_name:
            try:
                client.caches.delete(name=cache_name)
            except Exception:
                pass

        nc = compute_costs(no_cache_stats)
        cc = compute_costs(cached_stats)
        savings = nc["cost_total"] - cc["cost_total"]
        savings_pct = (savings / nc["cost_total"] * 100) if nc["cost_total"] > 0 else 0

        all_results[scenario_name] = {
            "rounds": len(prompts),
            "no_cache": nc,
            "cached": cc,
            "savings": savings,
            "savings_pct": savings_pct,
        }

    # ── Final Report ──
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Scenario':<25} {'Rounds':>6} {'No Cache':>12} {'Cached':>12} {'Savings':>12} {'%':>8}")
    print("─" * 75)

    total_no_cache = 0
    total_cached = 0

    for name, r in all_results.items():
        total_no_cache += r["no_cache"]["cost_total"]
        total_cached += r["cached"]["cost_total"]
        print(
            f"{name:<25} {r['rounds']:>6} "
            f"${r['no_cache']['cost_total']:>10.4f} "
            f"${r['cached']['cost_total']:>10.4f} "
            f"${r['savings']:>10.4f} "
            f"{r['savings_pct']:>7.1f}%"
        )

    total_savings = total_no_cache - total_cached
    total_pct = (total_savings / total_no_cache * 100) if total_no_cache > 0 else 0

    print("─" * 75)
    print(
        f"{'TOTAL':<25} {'':>6} "
        f"${total_no_cache:>10.4f} "
        f"${total_cached:>10.4f} "
        f"${total_savings:>10.4f} "
        f"{total_pct:>7.1f}%"
    )

    # Detailed breakdown
    for name, r in all_results.items():
        nc = r["no_cache"]
        cc = r["cached"]
        print(f"\n{'─' * 60}")
        print(f"  {name} ({r['rounds']} rounds)")
        print(f"{'─' * 60}")
        print(f"  WITHOUT cache:")
        print(f"    Input tokens:  {nc['total_input']:>12,}  (${nc['cost_input_standard']:.4f})")
        print(f"    Output tokens: {nc['total_output']:>12,}  (${nc['cost_output']:.4f})")
        print(f"    Total cost:    {'':>12}   ${nc['cost_total']:.4f}")
        print(f"  WITH cache:")
        print(f"    Standard input:{cc['standard_input']:>12,}  (${cc['cost_input_standard']:.4f})")
        print(f"    Cached input:  {cc['total_cached']:>12,}  (${cc['cost_input_cached']:.4f})")
        print(f"    Output tokens: {cc['total_output']:>12,}  (${cc['cost_output']:.4f})")
        print(f"    Cache storage: {'':>12}   ${cc['cost_storage']:.4f}")
        print(f"    Total cost:    {'':>12}   ${cc['cost_total']:.4f}")
        print(f"  SAVINGS: ${r['savings']:.4f} ({r['savings_pct']:.1f}%)")

    # Monthly projection
    print(f"\n{'=' * 80}")
    print("MONTHLY PROJECTION (assuming 50 sessions/day)")
    print(f"{'=' * 80}")
    sessions_per_month = 50 * 30
    avg_no_cache = total_no_cache / len(all_results)
    avg_cached = total_cached / len(all_results)
    monthly_no_cache = avg_no_cache * sessions_per_month
    monthly_cached = avg_cached * sessions_per_month
    monthly_savings = monthly_no_cache - monthly_cached

    print(f"  Avg session cost (no cache):  ${avg_no_cache:.4f}")
    print(f"  Avg session cost (cached):    ${avg_cached:.4f}")
    print(f"  Monthly (no cache):           ${monthly_no_cache:,.2f}")
    print(f"  Monthly (cached):             ${monthly_cached:,.2f}")
    print(f"  Monthly savings:              ${monthly_savings:,.2f}")
    print(f"  Annual savings:               ${monthly_savings * 12:,.2f}")


if __name__ == "__main__":
    run_benchmark()
