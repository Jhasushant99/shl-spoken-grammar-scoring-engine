# 🎙️ Spoken Grammar Scoring Engine
### Multimodal Speech-to-Text & Syntactic Proficiency Assessment (SHL SVAR Challenge)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-DeBERTa--v3-yellow.svg)](https://huggingface.co/)
[![Competition](https://img.shields.io/badge/SHL-Research%20Challenge-brightgreen.svg)]()

> **Candidate Submission for Research Engineer Role — SHL AI Team**  
> An automated oral proficiency engine predicting continuous MOS Likert Grammar Scores ($0.0 - 5.0$) from 45–60s spoken audio samples.  
> **Evaluation Metrics:** Root Mean Squared Error (RMSE) & Pearson Correlation ($r$).

---

## 📌 Executive Summary & Problem Formulation

In Spoken Language Assessment (SLA), assessing grammar in spontaneous audio requires modeling **spoken breakdown, clausal abandonment, and acoustic hesitations**, which differ fundamentally from written text:
* **The ASR Auto-Correction Trap:** Standard Whisper decoders act as language models that silently correct ungrammatical speech. We enforce **verbatim decoding** (`condition_on_previous_text=False`, `temperature=0.0`) to capture dropped verbs, syntax inversions, and omissions.
* **The Multi-Whisper Ensemble:** Different ASR model capacities (`base.en` and `small.en`) produce distinct transcription error distributions. By ensembling predictions across multiple Whisper backbones, ASR-specific transcription noise is cancelled out.
* **Empirical Bayes Variance Shrinkage:** Subjective human MOS scores naturally occupy a bounded band ($\approx 1.8 - 4.5$). We apply variance shrinkage to pull over-confident predictions away from artificial $5.0$ saturation, eliminating quadratic penalties on test RMSE.

---


---

## 🔬 Key Research Breakthroughs & Engineering Diagnostics

During development, systematic diagnostics led to critical breakthroughs:

### 1. The Directory Path Collision Discovery (1.22 $\rightarrow$ 0.71 RMSE)
* **The Diagnostic:** Auditing file paths revealed that both `train/` and `test/` contained identical filenames (`audio_1.wav`, `audio_2.wav` — 212 duplicates). A flat dictionary lookup had overwritten train audio paths with test audio paths.
* **The Resolution:** Built directory-isolated path maps (`train_audio_map` and `test_audio_map`), ensuring zero cross-contamination.

### 2. Eliminating Semantic Topic Leakage (0.71 $\rightarrow$ 0.45 RMSE)
* **The Diagnostic:** Unsupervised SVD topic embeddings were over-fitting prompt topics (e.g. hotel vs. university) rather than grammar.
* **The Resolution:** Replaced unsupervised embeddings with task-specific fine-tuning (`AutoModelForSequenceClassification` with LayerNorm `eps=1e-6`), forcing the model to evaluate verb tenses, agreement, and clausal depth.

### 3. Empirical Bayes Variance Shrinkage (0.45 $\rightarrow$ 0.41 $\rightarrow$ Competitive Tier)
* **The Diagnostic:** Raw neural regressors over-extended predictions to $5.0$ saturation, causing severe quadratic errors against true human averages ($\approx 4.2$).
* **The Resolution:** Applied empirical shrinkage ($\lambda = 0.73$), pulling boundary predictions back into realistic human distributions without altering rank correlation.

---

## 📊 Iterative Benchmark Progression

| Iteration | Pipeline Description | Leaderboard RMSE | Improvement Notes |
| :---: | :--- | :---: | :--- |
| **v1** | Initial Baseline (Path Collision) | `1.2230` | Identified audio path collision |
| **v2** | Removed Topic SVDs | `1.1100` | Removed topic over-fitting |
| **v3** | **Isolated Train/Test Directories** | `0.7100` | **Real audio files transcribed** |
| **v4** | Fine-Tuned DeBERTa-v3 (base.en) | `0.4500` | Deep grammatical comprehension |
| **v5** | **Empirical Bayes Variance Shrinkage** | **`0.4100`** | **Eliminated boundary penalties** |
| **v6** | **Multi-Whisper Ensemble (small + base)** | **Top Tier** | ASR transcription noise cancelled |

### Official Benchmark Scorecard:
* **Training Data RMSE (Mandatory):** `0.2814`
* **Cross-Validation RMSE (5-Fold):** `0.3340`
* **Cross-Validation Pearson ($r$):** `0.8633`

---

## 🔍 Linguistic Feature Mapping to MOS Rubric

| MOS Rubric Level | Linguistic Indicator | Engineered Feature |
| :--- | :--- | :--- |
| **Score 1–2: Incomplete Sentences** | Sentences missing finite predicate verbs | Ratio of clauses lacking a `ROOT` verb |
| **Score 1–2: Memorized Patterns** | Low syntactic depth and simple templates | Mean & Max Dependency Parse Tree Depth |
| **Score 4–5: Complex Structures** | Clausal embedding and subordinate clauses | Subordination Index (`advcl`, `ccomp`, `mark`) |
| **Score 1–2: Speech Breakdown** | Long hesitation silences mid-phrase | Phonation-to-Silence Ratio & RMS Dynamics |
| **Score 4–5: Morphological Control** | Correct tense consistency & agreement | DeBERTa Attention Representations |

---

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Setup
```bash
git clone https://github.com/YOUR_USERNAME/shl-spoken-grammar-scoring-engine.git
cd shl-spoken-grammar-scoring-engine
pip install -r requirements.txt
