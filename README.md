# Spoken Grammar Scoring Engine (SHL Research Challenge)

An automated multimodal assessment engine predicting continuous MOS Likert Grammar Scores (0.0 – 5.0) from spoken audio samples (45–60s).

## Architecture Overview
1. **Verbatim ASR Decoding (`faster-whisper`):** Decodes speech with disabled language-model conditioning (`condition_on_previous_text=False`) to preserve genuine spoken disfluencies and grammatical slips without hallucinated corrections.
2. **Syntactic & Grammatical Modeling (`spaCy`):** Extracts dependency parse tree depth, clausal subordination index, incomplete sentence ratios, and lexical diversity (TTR) aligned directly with the MOS rubric.
3. **Acoustic Prosody & Fluency (`librosa`):** Quantifies phonation time ratio, mid-clause pause counts, and RMS energy dynamics.
4. **Contextual Representations:** Semantic feature compression via DeBERTa-v3.
5. **Stratified 5-Fold Regularized Ensemble & Metric Calibration:** Gradient boosted ensemble (LightGBM + CatBoost) with Nelder-Mead post-processing optimizing both RMSE and Pearson correlation ($r$).

## How to Run
1. Open the notebook in `notebooks/spoken_grammar_scoring.ipynb` or run in Kaggle with GPU enabled.
2. Install dependencies: `pip install -r requirements.txt`.
