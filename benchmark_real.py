"""Benchmark: Real multi-round agent conversation with caching ON vs OFF.

Uses the agent_core Agent class with tools that return large responses
to grow context quickly. Runs the same 12+ prompts twice — once without
caching and once with caching — then prints a comparison table.

Usage:
    conda run -n base python -u benchmark_real.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── Load environment ──
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))
ldia_env = Path(__file__).resolve().parent.parent / "ldia" / ".env"
if ldia_env.exists():
    load_dotenv(ldia_env, override=True)

# Ensure GOOGLE_PROJECT_ID is set
if not os.environ.get("GOOGLE_PROJECT_ID"):
    os.environ["GOOGLE_PROJECT_ID"] = "exp001-429822"

from agent_core.agents.base import Agent
from agent_core.core.events import EventType, get_event_bus

# ── Pricing (gemini-3-pro-preview, <= 200K context) ──
PRICE_INPUT_STANDARD = 2.00   # per million tokens
PRICE_INPUT_CACHED   = 0.20   # per million tokens
PRICE_OUTPUT         = 12.00  # per million tokens


# ═══════════════════════════════════════════════════════════════════════
# Large deterministic tool responses
# ═══════════════════════════════════════════════════════════════════════

DATASET_PROFILE = {
    "dataset_name": "sensor_telemetry_v3",
    "source": "Industrial IoT Gateway — Plant Floor Alpha",
    "collection_period": "2024-01-01 to 2024-12-31",
    "total_rows": 8_742_391,
    "total_columns": 42,
    "file_size_bytes": 3_217_894_502,
    "columns": {
        "timestamp":          {"dtype": "datetime64[ns]", "nulls": 0,   "unique": 8_742_391, "min": "2024-01-01T00:00:00", "max": "2024-12-31T23:59:59"},
        "sensor_id":          {"dtype": "string",         "nulls": 0,   "unique": 256,       "top": "SNR-A-0042",          "freq": 34_150},
        "temperature_c":      {"dtype": "float64",        "nulls": 312, "mean": 72.43,  "std": 8.91,   "min": -12.7,  "max": 198.4,  "q25": 66.1,  "q50": 72.3,  "q75": 78.8,  "skew": 0.42,  "kurtosis": 3.87},
        "humidity_pct":       {"dtype": "float64",        "nulls": 187, "mean": 45.21,  "std": 12.34,  "min": 5.2,    "max": 99.8,   "q25": 36.0,  "q50": 44.8,  "q75": 54.1,  "skew": 0.18,  "kurtosis": 2.63},
        "pressure_kpa":       {"dtype": "float64",        "nulls": 45,  "mean": 101.32, "std": 1.47,   "min": 95.1,   "max": 108.9,  "q25": 100.3, "q50": 101.3, "q75": 102.3, "skew": -0.05, "kurtosis": 3.12},
        "vibration_mm_s":     {"dtype": "float64",        "nulls": 892, "mean": 2.87,   "std": 1.53,   "min": 0.01,   "max": 48.92,  "q25": 1.78,  "q50": 2.61,  "q75": 3.62,  "skew": 3.41,  "kurtosis": 22.7},
        "current_amps":       {"dtype": "float64",        "nulls": 67,  "mean": 14.82,  "std": 3.21,   "min": 0.0,    "max": 42.1,   "q25": 12.5,  "q50": 14.7,  "q75": 17.0,  "skew": 0.31,  "kurtosis": 3.44},
        "voltage_v":          {"dtype": "float64",        "nulls": 23,  "mean": 229.8,  "std": 5.12,   "min": 198.3,  "max": 258.7,  "q25": 226.4, "q50": 229.9, "q75": 233.1, "skew": -0.02, "kurtosis": 2.98},
        "rpm":                {"dtype": "float64",        "nulls": 134, "mean": 1472.3, "std": 312.8,  "min": 0.0,    "max": 3600.0, "q25": 1250.0,"q50": 1470.0,"q75": 1700.0,"skew": 0.08,  "kurtosis": 2.54},
        "acoustic_db":        {"dtype": "float64",        "nulls": 1023,"mean": 68.4,   "std": 11.2,   "min": 32.1,   "max": 118.7,  "q25": 60.5,  "q50": 67.8,  "q75": 75.9,  "skew": 0.52,  "kurtosis": 3.21},
        "power_kw":           {"dtype": "float64",        "nulls": 89,  "mean": 3.41,   "std": 1.12,   "min": 0.0,    "max": 12.8,   "q25": 2.64,  "q50": 3.38,  "q75": 4.12,  "skew": 0.45,  "kurtosis": 3.67},
        "oil_level_pct":      {"dtype": "float64",        "nulls": 2341,"mean": 78.2,   "std": 14.3,   "min": 12.0,   "max": 100.0,  "q25": 69.0,  "q50": 80.0,  "q75": 89.0,  "skew": -0.64, "kurtosis": 2.81},
        "coolant_temp_c":     {"dtype": "float64",        "nulls": 567, "mean": 38.7,   "std": 6.82,   "min": 8.2,    "max": 92.4,   "q25": 34.1,  "q50": 38.4,  "q75": 43.0,  "skew": 0.38,  "kurtosis": 4.12},
        "bearing_temp_c":     {"dtype": "float64",        "nulls": 789, "mean": 54.3,   "std": 9.45,   "min": 18.0,   "max": 142.0,  "q25": 48.0,  "q50": 53.8,  "q75": 60.2,  "skew": 0.71,  "kurtosis": 4.89},
        "load_pct":           {"dtype": "float64",        "nulls": 156, "mean": 67.8,   "std": 18.9,   "min": 0.0,    "max": 105.0,  "q25": 55.0,  "q50": 68.0,  "q75": 82.0,  "skew": -0.21, "kurtosis": 2.43},
        "anomaly_flag":       {"dtype": "int8",           "nulls": 0,   "unique": 2,     "value_counts": {"0": 8_412_000, "1": 330_391}},
        "maintenance_window": {"dtype": "bool",           "nulls": 0,   "unique": 2,     "value_counts": {"False": 8_200_000, "True": 542_391}},
        "zone":               {"dtype": "category",       "nulls": 0,   "unique": 8,     "categories": ["A1","A2","A3","B1","B2","B3","C1","C2"]},
        "machine_type":       {"dtype": "category",       "nulls": 0,   "unique": 12,    "categories": ["compressor","pump","motor","turbine","conveyor","mixer","dryer","press","lathe","grinder","welder","crane"]},
    },
    "correlations_with_anomaly_flag": {
        "vibration_mm_s":  0.621,
        "bearing_temp_c":  0.534,
        "acoustic_db":     0.489,
        "temperature_c":   0.387,
        "current_amps":    0.312,
        "coolant_temp_c":  0.278,
        "load_pct":        0.245,
        "power_kw":        0.198,
        "rpm":             0.134,
        "oil_level_pct":  -0.312,
        "humidity_pct":   -0.087,
        "pressure_kpa":   -0.043,
        "voltage_v":      -0.021,
    },
    "missing_data_summary": {
        "total_missing_cells": 6_625,
        "columns_with_nulls": 14,
        "columns_fully_complete": 28,
        "worst_column": "oil_level_pct (2341 nulls, 0.027%)",
        "pattern": "Missing values are NOT random — concentrated during maintenance windows and sensor recalibration periods.",
    },
    "temporal_patterns": {
        "seasonality": "Strong diurnal pattern in temperature, humidity; weekly pattern in load_pct",
        "trend": "Gradual increase in vibration_mm_s over 12-month period (potential bearing degradation)",
        "anomaly_clusters": "72% of anomaly_flag=1 events occur between 02:00-06:00 UTC (night shift)",
        "data_gaps": "3 gaps >1hr: 2024-03-15 (planned outage), 2024-07-22 (network failure), 2024-11-01 (sensor firmware update)",
    },
}

ANALYSIS_RESULT = {
    "analysis_type": "Multivariate Correlation & Anomaly Pattern Analysis",
    "timestamp": "2024-12-15T14:32:00Z",
    "execution_time_seconds": 47.3,
    "methodology": "Pearson correlation matrix, Spearman rank correlation for non-linear relationships, PCA for dimensionality reduction, Isolation Forest for anomaly detection validation",
    "findings": {
        "primary_correlations": [
            {"pair": ("vibration_mm_s", "bearing_temp_c"), "pearson": 0.78, "spearman": 0.81, "interpretation": "Strong positive — vibration causes friction heating in bearings. This is the primary failure cascade path."},
            {"pair": ("vibration_mm_s", "acoustic_db"),    "pearson": 0.72, "spearman": 0.69, "interpretation": "Strong positive — mechanical vibration produces audible noise. Acoustic monitoring validates vibration sensors."},
            {"pair": ("current_amps", "load_pct"),         "pearson": 0.89, "spearman": 0.87, "interpretation": "Very strong positive — electrical current directly tracks mechanical load. Expected physical relationship."},
            {"pair": ("temperature_c", "coolant_temp_c"),  "pearson": 0.65, "spearman": 0.63, "interpretation": "Moderate positive — ambient temperature influences coolant effectiveness. Seasonal adjustment needed."},
            {"pair": ("rpm", "power_kw"),                  "pearson": 0.71, "spearman": 0.68, "interpretation": "Strong positive — power consumption scales with rotational speed. Non-linear at high RPM due to aerodynamic drag."},
            {"pair": ("oil_level_pct", "bearing_temp_c"),  "pearson": -0.54, "spearman": -0.58, "interpretation": "Moderate negative — low oil leads to increased bearing friction and heat. Critical maintenance indicator."},
        ],
        "pca_analysis": {
            "components_explaining_95pct_variance": 6,
            "pc1_explained_variance": 0.312,
            "pc1_top_loadings": {"vibration_mm_s": 0.42, "bearing_temp_c": 0.38, "acoustic_db": 0.35, "current_amps": 0.28},
            "pc2_explained_variance": 0.198,
            "pc2_top_loadings": {"temperature_c": 0.51, "humidity_pct": -0.44, "coolant_temp_c": 0.39},
            "pc3_explained_variance": 0.142,
            "pc3_top_loadings": {"rpm": 0.56, "power_kw": 0.48, "load_pct": 0.41},
        },
        "anomaly_detection_validation": {
            "isolation_forest_auc": 0.847,
            "precision_at_10pct_recall": 0.923,
            "top_features_by_importance": [
                {"feature": "vibration_mm_s",  "importance": 0.234, "rank": 1},
                {"feature": "bearing_temp_c",  "importance": 0.198, "rank": 2},
                {"feature": "acoustic_db",     "importance": 0.156, "rank": 3},
                {"feature": "oil_level_pct",   "importance": 0.121, "rank": 4},
                {"feature": "current_amps",    "importance": 0.098, "rank": 5},
                {"feature": "temperature_c",   "importance": 0.076, "rank": 6},
                {"feature": "coolant_temp_c",  "importance": 0.054, "rank": 7},
                {"feature": "load_pct",        "importance": 0.032, "rank": 8},
                {"feature": "rpm",             "importance": 0.019, "rank": 9},
                {"feature": "power_kw",        "importance": 0.012, "rank": 10},
            ],
        },
        "temporal_decomposition": {
            "stl_decomposition": "Applied to vibration_mm_s with period=24 (hourly data, daily cycle)",
            "trend_slope_per_month": 0.037,
            "seasonal_amplitude": 0.42,
            "residual_std": 0.89,
            "changepoints_detected": [
                {"date": "2024-04-12", "metric": "vibration_mm_s", "shift": "+0.31 mm/s", "probable_cause": "Bearing B-142 replacement"},
                {"date": "2024-08-03", "metric": "bearing_temp_c", "shift": "+4.2°C",     "probable_cause": "Cooling system partial failure"},
                {"date": "2024-10-19", "metric": "acoustic_db",    "shift": "+6.8 dB",    "probable_cause": "Loose mounting bolt on compressor C-7"},
            ],
        },
    },
    "recommendations": [
        "Deploy real-time vibration threshold alerting at 2σ above rolling 7-day mean",
        "Implement predictive maintenance model using top-5 features from Isolation Forest",
        "Add oil level monitoring integration — currently 2341 missing values indicate sensor gaps",
        "Investigate night-shift anomaly clustering — possible operator behavior or environmental factor",
        "Consider seasonal normalization for temperature-dependent features before model training",
    ],
}

REPORT_TEXT = """
═══════════════════════════════════════════════════════════════════════════════
                    INDUSTRIAL IOT SENSOR ANALYSIS REPORT
                    Plant Floor Alpha — Annual Review 2024
═══════════════════════════════════════════════════════════════════════════════

1. EXECUTIVE SUMMARY
────────────────────
This report presents findings from a comprehensive analysis of 8.7 million
telemetry records collected from 256 sensors across Plant Floor Alpha during
the 2024 calendar year. The analysis covers data quality assessment,
multivariate correlation analysis, anomaly detection validation, and
predictive maintenance recommendations.

Key Finding: Vibration-bearing temperature cascade is the primary failure
pathway, accounting for 62.1% of anomaly events. A predictive model using
the top 5 features achieves AUC 0.847, with precision of 92.3% when
targeting the top 10% of predictions.

2. DATA QUALITY ASSESSMENT
──────────────────────────
Total Records: 8,742,391 across 42 columns
Missing Data:  6,625 cells (0.0076% of all cells)
  - oil_level_pct: 2,341 nulls (highest) — sensor calibration gaps
  - acoustic_db:   1,023 nulls — microphone saturation events
  - vibration_mm_s:  892 nulls — sensor communication timeouts
  - bearing_temp_c:  789 nulls — thermocouple intermittent failures
  - coolant_temp_c:  567 nulls — flow sensor air bubbles

Missing data is NOT random (MNAR pattern). 78% of missing values occur
during maintenance windows or sensor recalibration periods. Recommend
forward-fill with exponential decay for gaps < 5 minutes, and explicit
NA encoding for longer gaps.

3. CORRELATION ANALYSIS
───────────────────────
Primary Failure Cascade (vibration → heat → failure):
  vibration_mm_s ↔ bearing_temp_c: r=0.78 (Pearson), ρ=0.81 (Spearman)
  vibration_mm_s ↔ acoustic_db:    r=0.72, ρ=0.69
  bearing_temp_c ↔ oil_level_pct:  r=-0.54, ρ=-0.58

Operational Relationships:
  current_amps ↔ load_pct:  r=0.89 (expected — Ohm's law)
  rpm ↔ power_kw:           r=0.71 (non-linear at high RPM)
  temperature_c ↔ coolant:  r=0.65 (seasonal dependency)

4. ANOMALY DETECTION
────────────────────
Anomaly Rate: 3.78% (330,391 flagged events out of 8,742,391)
Temporal Distribution:
  - Night shift (02:00-06:00 UTC): 72% of anomalies
  - Day shift (06:00-14:00 UTC):   18% of anomalies
  - Evening shift (14:00-22:00):   10% of anomalies

This 7.2x overrepresentation during night shift warrants investigation.
Possible factors: reduced staffing, environmental conditions (temperature
drops), different operational patterns.

Isolation Forest Validation:
  - AUC: 0.847
  - Precision at 10% recall: 0.923
  - Feature importance (top 5):
    1. vibration_mm_s  (23.4%)
    2. bearing_temp_c  (19.8%)
    3. acoustic_db     (15.6%)
    4. oil_level_pct   (12.1%)
    5. current_amps    (9.8%)

5. PRINCIPAL COMPONENT ANALYSIS
───────────────────────────────
6 components explain 95% of variance in the 14 continuous features.

PC1 (31.2%): "Mechanical Stress" — vibration, bearing temp, acoustic
PC2 (19.8%): "Environmental" — ambient temp, humidity, coolant temp
PC3 (14.2%): "Operational Load" — RPM, power, load percentage
PC4 (11.1%): "Electrical" — current, voltage
PC5 (9.8%):  "Lubrication" — oil level, bearing temp (inverse)
PC6 (8.9%):  "Pressure-Humidity Interaction"

6. TEMPORAL PATTERNS
────────────────────
STL Decomposition of vibration_mm_s:
  - Trend: +0.037 mm/s per month (bearing degradation)
  - Seasonal amplitude: 0.42 mm/s (24-hour cycle)
  - Residual std: 0.89 mm/s

Changepoints Detected:
  2024-04-12: vibration_mm_s +0.31 mm/s (Bearing B-142 replacement)
  2024-08-03: bearing_temp_c +4.2°C (Cooling system partial failure)
  2024-10-19: acoustic_db +6.8 dB (Loose mounting bolt, compressor C-7)

7. PREDICTIVE MAINTENANCE MODEL
────────────────────────────────
Gradient Boosted Trees (XGBoost):
  - Predicts anomaly_flag 6 hours in advance
  - AUC: 0.847 | Precision@10%: 0.923 | F1: 0.812
  - Estimated annual savings: $2.4M (prevented unplanned downtime)

Feature Set (ranked by SHAP values):
  1. vibration_mm_s (rolling 1hr mean)
  2. bearing_temp_c (rate of change, last 30min)
  3. acoustic_db (rolling 1hr std)
  4. oil_level_pct (current value)
  5. vibration_mm_s × bearing_temp_c (interaction term)

8. RECOMMENDATIONS
──────────────────
  [HIGH]   Deploy real-time vibration threshold alerting at 2σ above 7-day rolling mean
  [HIGH]   Implement predictive model using top-5 SHAP features
  [MEDIUM] Fix oil level sensor gaps (2,341 missing values)
  [MEDIUM] Investigate night-shift anomaly clustering — 7.2x overrepresentation
  [MEDIUM] Add seasonal normalization for temperature-dependent features
  [LOW]    Evaluate LSTM-based sequence model for longer prediction horizons
  [LOW]    Consider adding vibration frequency spectrum analysis (FFT features)

9. APPENDIX — STATISTICAL TABLES
─────────────────────────────────
Table A1: Complete Descriptive Statistics (all 14 continuous features)
┌───────────────────┬──────────┬────────┬────────┬─────────┬────────┬────────┬────────┬────────┐
│ Feature           │    Count │   Mean │    Std │     Min │    25% │    50% │    75% │    Max │
├───────────────────┼──────────┼────────┼────────┼─────────┼────────┼────────┼────────┼────────┤
│ temperature_c     │ 8742079  │  72.43 │   8.91 │  -12.70 │  66.10 │  72.30 │  78.80 │ 198.40 │
│ humidity_pct      │ 8742204  │  45.21 │  12.34 │    5.20 │  36.00 │  44.80 │  54.10 │  99.80 │
│ pressure_kpa      │ 8742346  │ 101.32 │   1.47 │   95.10 │ 100.30 │ 101.30 │ 102.30 │ 108.90 │
│ vibration_mm_s    │ 8741499  │   2.87 │   1.53 │    0.01 │   1.78 │   2.61 │   3.62 │  48.92 │
│ current_amps      │ 8742324  │  14.82 │   3.21 │    0.00 │  12.50 │  14.70 │  17.00 │  42.10 │
│ voltage_v         │ 8742368  │ 229.80 │   5.12 │  198.30 │ 226.40 │ 229.90 │ 233.10 │ 258.70 │
│ rpm               │ 8742257  │1472.30 │ 312.80 │    0.00 │1250.00 │1470.00 │1700.00 │3600.00 │
│ acoustic_db       │ 8741368  │  68.40 │  11.20 │   32.10 │  60.50 │  67.80 │  75.90 │ 118.70 │
│ power_kw          │ 8742302  │   3.41 │   1.12 │    0.00 │   2.64 │   3.38 │   4.12 │  12.80 │
│ oil_level_pct     │ 8740050  │  78.20 │  14.30 │   12.00 │  69.00 │  80.00 │  89.00 │ 100.00 │
│ coolant_temp_c    │ 8741824  │  38.70 │   6.82 │    8.20 │  34.10 │  38.40 │  43.00 │  92.40 │
│ bearing_temp_c    │ 8741602  │  54.30 │   9.45 │   18.00 │  48.00 │  53.80 │  60.20 │ 142.00 │
│ load_pct          │ 8742235  │  67.80 │  18.90 │    0.00 │  55.00 │  68.00 │  82.00 │ 105.00 │
│ power_factor      │ 8742391  │   0.87 │   0.06 │    0.42 │   0.84 │   0.88 │   0.91 │   0.99 │
└───────────────────┴──────────┴────────┴────────┴─────────┴────────┴────────┴────────┴────────┘

Table A2: Anomaly Rate by Machine Type
┌──────────────┬───────────┬──────────┬──────────────┐
│ Machine Type │    Count  │ Anomalies│ Anomaly Rate │
├──────────────┼───────────┼──────────┼──────────────┤
│ compressor   │ 1,245,341 │   62,847 │        5.05% │
│ pump         │   987,234 │   43,210 │        4.38% │
│ motor        │   876,543 │   35,432 │        4.04% │
│ turbine      │   765,432 │   34,876 │        4.56% │
│ conveyor     │   654,321 │   21,345 │        3.26% │
│ mixer        │   543,210 │   18,765 │        3.45% │
│ dryer        │   432,109 │   15,432 │        3.57% │
│ press        │   876,543 │   30,876 │        3.52% │
│ lathe        │   654,321 │   22,345 │        3.41% │
│ grinder      │   543,210 │   19,876 │        3.66% │
│ welder       │   432,109 │   14,321 │        3.31% │
│ crane        │   732,018 │   31,066 │        4.24% │
└──────────────┴───────────┴──────────┴──────────────┘

END OF REPORT
═══════════════════════════════════════════════════════════════════════════════
"""


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Agent subclass
# ═══════════════════════════════════════════════════════════════════════

class BenchmarkAgent(Agent):
    """Agent with large-response tools for caching benchmarks."""

    name = "benchmark"
    system_prompt = (
        "You are an industrial IoT data analyst specializing in predictive maintenance. "
        "You have access to tools that profile sensor telemetry datasets, run analyses, "
        "and generate reports. Always use the tools when asked to profile, analyze, or "
        "report — do not make up data. Refer to previous tool results in your reasoning. "
        "When the user asks you to analyze something, call the appropriate tool first, "
        "then explain the results. Be thorough and reference specific numbers."
    )

    # Will be set per session
    ENABLE_CACHING = True

    # No lifecycle events during benchmark (cleaner output)
    emit_lifecycle_events = True
    emit_tool_events = False

    def __init__(self, enable_caching: bool = True):
        # Set class-level before super().__init__ reads it
        self.__class__.ENABLE_CACHING = enable_caching
        super().__init__()

        # Register tools (methods with type hints and docstrings)
        self.register_tool(self.get_dataset_profile)
        self.register_tool(self.execute_analysis)
        self.register_tool(self.get_report)

        # Per-round usage capture
        self._round_usages: list[dict] = []

    # ── Tools ──

    def get_dataset_profile(self, dataset_name: str = "sensor_telemetry_v3") -> dict:
        """Profile a sensor telemetry dataset, returning column statistics, correlations, and data quality info.

        Args:
            dataset_name: Name of the dataset to profile.

        Returns:
            A detailed profile including column stats, correlations, missing data patterns, and temporal info.
        """
        return DATASET_PROFILE

    def execute_analysis(self, analysis_type: str = "correlation", focus_features: str = "all") -> dict:
        """Run a multivariate analysis on the sensor dataset including correlations, PCA, and anomaly detection.

        Args:
            analysis_type: Type of analysis — correlation, pca, anomaly_detection, or temporal.
            focus_features: Comma-separated feature names to focus on, or 'all'.

        Returns:
            Analysis results including correlations, PCA loadings, feature importances, and changepoints.
        """
        return ANALYSIS_RESULT

    def get_report(self, report_type: str = "full", sections: str = "all") -> dict:
        """Generate a formatted analysis report covering data quality, correlations, anomalies, and recommendations.

        Args:
            report_type: Report type — full, summary, or executive.
            sections: Comma-separated section names or 'all'.

        Returns:
            Formatted report text with all analysis findings.
        """
        return {"report": REPORT_TEXT, "word_count": len(REPORT_TEXT.split()), "format": "plain_text"}


# ═══════════════════════════════════════════════════════════════════════
# Usage metadata capture via event bus
# ═══════════════════════════════════════════════════════════════════════

class UsageCapture:
    """Captures AGENT_END events to extract per-round token usage including cached tokens."""

    def __init__(self):
        self.events: list[dict] = []
        self._bus = get_event_bus()
        self._bus.subscribe(self._on_event)

    def _on_event(self, event):
        if event.type == EventType.AGENT_END:
            self.events.append({
                "token_usage": event.details.get("token_usage", {}),
                "status": str(event.status),
                "timestamp": event.timestamp,
            })

    def pop_latest(self) -> dict:
        """Pop the most recent AGENT_END event's token_usage."""
        if self.events:
            return self.events.pop()
        return {}

    def clear(self):
        self.events.clear()

    def unsubscribe(self):
        self._bus.unsubscribe(self._on_event)


# ═══════════════════════════════════════════════════════════════════════
# Monkey-patch to capture cached_content_token_count from usage_metadata
# ═══════════════════════════════════════════════════════════════════════
#
# The base Agent._run_with_function_loop accumulates prompt_tokens and
# completion_tokens but discards cached_content_token_count.  We patch
# it to also track that field.

_original_run_with_function_loop = Agent._run_with_function_loop

def _patched_run_with_function_loop(self, contents, config, contents_offset=0):
    """Patched version that also captures cached_content_token_count."""
    from google.genai import types as _types

    iteration = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0

    while iteration < self.MAX_ITERATIONS:
        iteration += 1

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents[contents_offset:],
            config=config,
        )

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            total_prompt_tokens += getattr(meta, "prompt_token_count", 0) or 0
            total_completion_tokens += getattr(meta, "candidates_token_count", 0) or 0
            cached = getattr(meta, "cached_content_token_count", 0) or 0
            total_cached_tokens += cached

        token_usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "cached_tokens": total_cached_tokens,
        }

        if not response.candidates or not response.candidates[0].content:
            return response.text or "", token_usage

        model_content = response.candidates[0].content

        text_parts = []
        function_calls = []
        for part in model_content.parts:
            if hasattr(part, "function_call") and part.function_call:
                function_calls.append(part.function_call)
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        # Emit intermediate text if present alongside tool calls
        if text_parts and function_calls:
            combined_text = "\n".join(text_parts)
            self.on_model_thinking(combined_text)
            if self.emit_lifecycle_events:
                from agent_core.core.events import EventType as _ET, emit_event as _emit
                _emit(
                    _ET.MODEL_THINKING,
                    agent=self.instance_id,
                    agent_type=self.name,
                    parent_agent=self._parent_agent,
                    wave_id=self._wave_id,
                    details={"text": combined_text},
                )

        if not function_calls:
            contents.append(model_content)
            self._save_history()
            return response.text or "", token_usage

        contents.append(model_content)
        self._save_history()

        results = self._execute_tools_parallel(function_calls)

        function_response_content = self._build_function_response_content(results)
        contents.append(function_response_content)
        self._save_history()

        # Emit context update
        if self.emit_lifecycle_events:
            from agent_core.core.events import EventType as _ET, EventStatus as _ES, emit_event as _emit
            context_tokens = self._count_context_tokens()
            _emit(
                _ET.CONTEXT_UPDATE,
                agent=self.instance_id,
                agent_type=self.name,
                parent_agent=self._parent_agent,
                wave_id=self._wave_id,
                details={"context_tokens": context_tokens},
            )

    token_usage = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "cached_tokens": total_cached_tokens,
    }
    return f"[Max iterations ({self.MAX_ITERATIONS}) reached]", token_usage

# Apply monkey-patch
Agent._run_with_function_loop = _patched_run_with_function_loop


# ═══════════════════════════════════════════════════════════════════════
# Prompts — 14 rounds of data analysis conversation
# ═══════════════════════════════════════════════════════════════════════

PROMPTS = [
    # Round 1 — invoke get_dataset_profile
    "Profile the sensor telemetry dataset using the get_dataset_profile tool. I need to understand what we're working with.",

    # Round 2 — invoke execute_analysis
    "Now run a correlation analysis on the dataset. Use the execute_analysis tool with analysis_type='correlation'.",

    # Round 3 — reasoning over profile + analysis (no tool)
    "Looking at the profile and the correlation results, which sensors show the strongest relationship with the anomaly flag? Are the correlations consistent between Pearson and Spearman?",

    # Round 4 — invoke get_report
    "Generate a full analysis report using the get_report tool. I want to see everything consolidated.",

    # Round 5 — reasoning over all results
    "Based on the report, profile, and analysis results — summarize the primary failure cascade. Explain the vibration-to-bearing-temperature pathway with specific numbers.",

    # Round 6 — invoke execute_analysis again
    "Run a deeper analysis focusing specifically on vibration_mm_s and bearing_temp_c. Use execute_analysis with focus_features='vibration_mm_s,bearing_temp_c'.",

    # Round 7 — reasoning over everything
    "The PCA results show 6 components explain 95% of variance. How does each component map to a physical system in the plant? Reference the loadings from the analysis.",

    # Round 8 — invoke get_dataset_profile again
    "Re-profile the dataset to check the missing data patterns more carefully. Use get_dataset_profile.",

    # Round 9 — reasoning about data quality
    "You now have two profile results and two analysis results in context. Compare the missing data patterns — which columns have the most nulls and why? What's the MNAR pattern about?",

    # Round 10 — invoke execute_analysis (temporal)
    "Run a temporal decomposition analysis. Use execute_analysis with analysis_type='temporal'.",

    # Round 11 — reasoning about temporal patterns
    "The temporal analysis detected 3 changepoints. Cross-reference those changepoints with the anomaly rate by machine type from the report. Which machines were likely affected by each changepoint?",

    # Round 12 — invoke get_report (summary)
    "Generate a summary report using get_report with report_type='summary'. I want the key takeaways.",

    # Round 13 — synthesis
    "Looking at ALL the data we've collected — profiles, analyses, reports, temporal patterns — what are the top 3 most actionable insights? Be specific with numbers and reference the tool results.",

    # Round 14 — final reasoning (longest context)
    "If we deploy the predictive maintenance model described in the report, what ROI can we expect? Factor in the anomaly rate (3.78%), the AUC (0.847), the night-shift clustering (72%), and the estimated $2.4M annual savings. What assumptions are we making and what could go wrong?",
]


# ═══════════════════════════════════════════════════════════════════════
# Main benchmark logic
# ═══════════════════════════════════════════════════════════════════════

def run_session(label: str, enable_caching: bool, prompts: list[str]) -> list[dict]:
    """Run a full conversation session and return per-round stats."""

    print(f"\n{'─' * 80}")
    print(f"  SESSION: {label}")
    print(f"  Caching: {'ENABLED' if enable_caching else 'DISABLED'}")
    print(f"  Rounds:  {len(prompts)}")
    print(f"{'─' * 80}")

    # Clear global event bus between sessions
    bus = get_event_bus()
    bus.clear()

    capture = UsageCapture()
    agent = BenchmarkAgent(enable_caching=enable_caching)

    stats = []

    for i, prompt in enumerate(prompts):
        round_num = i + 1
        t0 = time.time()
        print(f"\n  Round {round_num:>2}/{len(prompts)}: {prompt[:70]}...")

        try:
            result = agent.run(prompt, temperature=0.4, max_output_tokens=4096)
        except Exception as e:
            print(f"           ERROR: {e}")
            stats.append({
                "round": round_num,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "history_len": len(agent._history),
                "context_tokens": 0,
                "cache_ready": False,
                "elapsed_s": time.time() - t0,
                "error": str(e),
            })
            continue

        elapsed = time.time() - t0

        # Get token usage from event
        event_data = capture.pop_latest()
        token_usage = event_data.get("token_usage", {})

        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        cached_tokens = token_usage.get("cached_tokens", 0)

        # Check cache state
        cache_ready = False
        if agent._cache_pipeline:
            cache_ready = agent._cache_pipeline.has_ready_cache

        history_len = len(agent._history)

        # Count context tokens (expensive but informative)
        context_tokens = agent._count_context_tokens()

        row = {
            "round": round_num,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "history_len": history_len,
            "context_tokens": context_tokens,
            "cache_ready": cache_ready,
            "elapsed_s": elapsed,
        }
        stats.append(row)

        cache_str = f" cached={cached_tokens:>7,}" if cached_tokens else "               "
        ready_str = " [CACHE READY]" if cache_ready else ""
        print(
            f"           in={prompt_tokens:>7,}  out={completion_tokens:>5,}"
            f"{cache_str}  ctx={context_tokens:>7,}  hist={history_len:>3}"
            f"  {elapsed:>5.1f}s{ready_str}"
        )

        # Brief preview of model response
        preview = result.replace("\n", " ")[:100]
        print(f"           >> {preview}...")

    # Cleanup
    capture.unsubscribe()
    agent.close()

    return stats


def compute_costs(stats: list[dict]) -> dict:
    """Compute dollar costs from stats."""
    total_input = sum(s["prompt_tokens"] for s in stats)
    total_output = sum(s["completion_tokens"] for s in stats)
    total_cached = sum(s["cached_tokens"] for s in stats)
    standard_input = total_input - total_cached

    cost_standard = (standard_input / 1_000_000) * PRICE_INPUT_STANDARD
    cost_cached = (total_cached / 1_000_000) * PRICE_INPUT_CACHED
    cost_output = (total_output / 1_000_000) * PRICE_OUTPUT

    return {
        "total_input": total_input,
        "total_output": total_output,
        "total_cached": total_cached,
        "standard_input": standard_input,
        "cost_standard": cost_standard,
        "cost_cached": cost_cached,
        "cost_output": cost_output,
        "cost_total": cost_standard + cost_cached + cost_output,
    }


def print_comparison_table(
    label_a: str, stats_a: list[dict],
    label_b: str, stats_b: list[dict],
):
    """Print a side-by-side comparison table."""

    print(f"\n{'=' * 100}")
    print("PER-ROUND COMPARISON")
    print(f"{'=' * 100}")
    hdr = (
        f"{'Round':>5} │"
        f" {'── ' + label_a + ' ──':^34s} │"
        f" {'── ' + label_b + ' ──':^42s} │"
        f" {'Savings':>8}"
    )
    print(hdr)
    sub = (
        f"{'':>5} │"
        f" {'Input':>8} {'Output':>8} {'Cached':>8} {'Time':>6} │"
        f" {'Input':>8} {'Output':>8} {'Cached':>8} {'Ready':>5} {'Time':>6} │"
        f" {'Tokens':>8}"
    )
    print(sub)
    print("─" * 100)

    for sa, sb in zip(stats_a, stats_b):
        r = sa["round"]
        cache_mark = "Y" if sb.get("cache_ready") else "-"
        saved = sa["prompt_tokens"] - (sb["prompt_tokens"] - sb["cached_tokens"]) if sb["cached_tokens"] else 0
        print(
            f"{r:>5} │"
            f" {sa['prompt_tokens']:>8,} {sa['completion_tokens']:>8,} {sa['cached_tokens']:>8,} {sa['elapsed_s']:>5.1f}s │"
            f" {sb['prompt_tokens']:>8,} {sb['completion_tokens']:>8,} {sb['cached_tokens']:>8,} {cache_mark:>5} {sb['elapsed_s']:>5.1f}s │"
            f" {saved:>8,}"
        )

    # Totals
    print("─" * 100)
    ca = compute_costs(stats_a)
    cb = compute_costs(stats_b)

    print(f"\n{'=' * 80}")
    print("COST SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Pricing: ${PRICE_INPUT_STANDARD}/M input, ${PRICE_INPUT_CACHED}/M cached input, ${PRICE_OUTPUT}/M output")
    print()

    for label, c in [(label_a, ca), (label_b, cb)]:
        print(f"  {label}:")
        print(f"    Standard input tokens: {c['standard_input']:>10,}  (${c['cost_standard']:.4f})")
        print(f"    Cached input tokens:   {c['total_cached']:>10,}  (${c['cost_cached']:.4f})")
        print(f"    Output tokens:         {c['total_output']:>10,}  (${c['cost_output']:.4f})")
        print(f"    TOTAL COST:            {'':>10}   ${c['cost_total']:.4f}")
        print()

    savings_dollars = ca["cost_total"] - cb["cost_total"]
    savings_pct = (savings_dollars / ca["cost_total"] * 100) if ca["cost_total"] > 0 else 0

    print(f"  {'─' * 40}")
    print(f"  SAVINGS: ${savings_dollars:.4f} ({savings_pct:.1f}%)")
    if ca["total_input"] > 0:
        effective_rate = cb["cost_total"] / ca["cost_total"] * 100
        print(f"  Effective cost ratio: {effective_rate:.1f}% of uncached cost")

    # Context growth table
    print(f"\n{'=' * 60}")
    print("CONTEXT GROWTH (tokens after each round)")
    print(f"{'=' * 60}")
    print(f"{'Round':>5} │ {label_a:>12} │ {label_b:>12} │ {'History':>7}")
    print("─" * 60)
    for sa, sb in zip(stats_a, stats_b):
        print(
            f"{sa['round']:>5} │ {sa['context_tokens']:>12,} │ {sb['context_tokens']:>12,} │ {sb['history_len']:>7}"
        )


def main():
    print("=" * 80)
    print("  AGENT_CORE CACHING BENCHMARK — Real Multi-Round Conversation")
    print("=" * 80)
    print(f"  Model:     gemini-3-pro-preview")
    print(f"  Project:   {os.environ.get('GOOGLE_PROJECT_ID', '???')}")
    print(f"  Rounds:    {len(PROMPTS)}")
    print(f"  Cache min: 32,768 tokens")
    print(f"  Pricing:   ${PRICE_INPUT_STANDARD}/M in, ${PRICE_INPUT_CACHED}/M cached, ${PRICE_OUTPUT}/M out")
    print(f"  Tools:     get_dataset_profile, execute_analysis, get_report")
    print("=" * 80)

    # Session A — no caching
    stats_no_cache = run_session("NO CACHE", enable_caching=False, prompts=PROMPTS)

    # Brief pause between sessions
    print("\n\n  ... pausing 5 seconds between sessions ...\n")
    time.sleep(5)

    # Session B — caching enabled
    stats_cached = run_session("CACHED", enable_caching=True, prompts=PROMPTS)

    # Comparison
    print_comparison_table("No Cache", stats_no_cache, "Cached", stats_cached)

    print("\n" + "=" * 80)
    print("  BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
