# CLMR

This repository contains CLMR, fixed experiment configurations, and our protocol adapters. No trained CLMR checkpoints are distributed; reproduce the results by training with the configurations below. Dataset contents, pretrained RoBERTa files, and third-party model implementations are obtained from their owners and are not redistributed in the source repository.

## 1. Environment

Run commands from this repository's root. The tested training environment is Python 3.11, PyTorch 2.6.0+cu126, Transformers 4.52.3, NumPy 1.26.4 and scikit-learn 1.5.2.

~~~powershell
conda env create -f environment.yml
conda activate clmr
~~~

For baseline input preparation and legacy dependencies:

~~~powershell
pip install -r requirements_baselines.txt
python -m spacy download en_core_web_sm
~~~

The automatic dependency parser is en_core_web_sm 3.8.0 with spaCy 3.8.15. Graph preprocessing can also be done in a separate environment, leaving the pinned training environment unchanged. If a GPU is unavailable, a complete large-data rerun will be slow. Different hardware/kernels may introduce small numerical variation even with the same seeds.

## 2. Obtain datasets and the pretrained backbone

Read the resource owners' terms before downloading. The links below are upstream sources, not copies hosted by this project.

| Resource | Download/source | Purpose |
|---|---|---|
| MisNet augmented data | [Authors' download](https://drive.google.com/file/d/1J15WFvekULkyzeHWJ2VV_HyJzmMPbznV/view), linked by [MisNet](https://github.com/SilasTHU/MisNet) | Exact VUA-18, standard VUA-Verb, MOH-X and TroFi inputs, including existing gloss and example fields |
| MelBERT TSV data | [Authors' download](https://drive.google.com/file/d/18hemekvUuOw-qkQhWv4F6qFnis6lkaog/view), linked by [MelBERT](https://github.com/jin530/MelBERT) | Original VUA-18 target ids and TSV ordering for baseline runs |
| ContrastWSD augmentation | [Authors' Kaggle data](https://www.kaggle.com/datasets/melzohbi/metaphor-detection-vua-wsd-augmented/data), linked by [ContrastWSD](https://github.com/melzohbi/ContrastWSD) | VUA-20 released train/test inputs and aligned contextual WSD fields |
| Original VUA-18/MOH-X/TroFi | [Sequential Metaphor Identification](https://github.com/RuiMao1988/Sequential-Metaphor-Identification) | Original benchmark provenance; not a substitute for the augmented inputs above |
| Original VUA-20 | [DeepMet](https://github.com/YU-NLPLab/DeepMet) | Original benchmark provenance |
| RoBERTa-base | [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) | Tokenizer and encoder initialization |

Unpack downloads locally into these ignored directories. The paths below refer to the directories containing the actual files, not to an arbitrary enclosing archive directory:

~~~text
downloads/misnet_data/
    VUA_All/{train,val,test}.csv
    VUA_Verb/{train,val,test}.csv
    MOH-X/MOH-X.csv
    TroFi/TroFi.csv
downloads/melbert_data/VUA18/{train,dev,test}.tsv
downloads/contrastwsd_data/
    VUA18/{train,test}.tsv
    VUAverb/{train,test}.tsv
    VUA20/{train,test}.tsv
~~~

Google Drive or Kaggle may require a browser/login; no credentials are bundled. RoBERTa can be downloaded with:

~~~powershell
python scripts/fetch_resources.py --backbone --accept-upstream-terms
~~~

Alternatively, place the authors' RoBERTa-base files under pretrained/roberta-base: config.json, vocab.json, merges.txt, tokenizer.json, and model.safetensors (or pytorch_model.bin). Training starts from the public RoBERTa-base initialization, not from a trained CLMR checkpoint.

## 3. Rebuild exactly the paper's data splits

~~~powershell
python scripts/prepare_data.py --misnet-data downloads/misnet_data --vua20-source downloads/contrastwsd_data/VUA20
python scripts/verify_resources.py --require-all
~~~

The script preserves the released VUA-18 and standard VUA-Verb splits. It preserves the existing knowledge fields rather than substituting freshly generated definitions. It builds MOH-X/TroFi stratified outer ten folds with split seed 20260815 and an inner validation fraction of 1/9 of the outer training fold (seed = outer seed + fold). These are instance-level folds, not duplicate-group-isolated folds.

VUA-20 uses the released test set unchanged and a fixed sentence-grouped validation holdout from released training: GroupShuffleSplit(seed=2026, validation fraction=0.0416). Its original preparation recipe stores the released definition as gloss and sentence as eg_sent; this source-specific mapping is retained for reproduction, not silently replaced with dictionary examples.

| Dataset | Train | Val | Test |
|---|---:|---:|---:|
| VUA-18 | 116622 | 38628 | 50175 |
| Standard VUA-Verb | 15516 | 1724 | 5873 |
| VUA-20 local fixed split | 153862 | 6292 | 22196 |
| MOH-X | Fold-specific | Fold-specific | 647 pooled outer-test targets |
| TroFi | Fold-specific | Fold-specific | 3737 pooled outer-test targets |

resources/data_manifest.json verifies 69 ordered split-content hashes, including the knowledge fields. Counts alone are not sufficient. 

## 4. Train CLMR under the fixed primary protocol

~~~powershell
python scripts/run_experiments.py --model clmr --dataset vua18
python scripts/run_experiments.py --model clmr --dataset vua_verb_standard
python scripts/run_experiments.py --model clmr --dataset vua20
python scripts/run_experiments.py --model clmr --dataset mohx
python scripts/run_experiments.py --model clmr --dataset trofi
~~~

Seeds are 13, 21 and 42 by default; use --seeds 13 for a diagnostic single-seed run. Checkpoints are selected using validation metaphor F1; the test set is evaluated only after checkpoint selection, with threshold 0.5. For MOH-X/TroFi, each seed concatenates all ten outer-test predictions before computing metrics. The reported spread is sample SD across seeds, not SD across folds.

Exact batch sizes, learning rates, class weights, sequence lengths and training budgets are in clmr/configs/. Scripts refuse to overwrite completed output directories. Choose a new --output-root for reruns. Training retains newly generated CLMR checkpoints; a complete ten-fold three-seed rerun will use substantial disk space.

For a parameter-matched ablation or the shuffled-reference intervention:

~~~powershell
python clmr/src/train.py --config clmr/configs/ablation_ce_only.json --seed 13 --output artifacts/ablations/ce_only/seed_13 --evaluate-test
python clmr/src/train.py --config clmr/configs/ablation_match.json --seed 13 --output artifacts/ablations/match/seed_13 --evaluate-test
python clmr/src/train.py --config clmr/configs/ablation_match_hard_negative.json --seed 13 --output artifacts/ablations/match_hn/seed_13 --evaluate-test
python clmr/src/train.py --config clmr/configs/clmr_vua18.json --seed 13 --output artifacts/controls/shuffled/seed_13 --knowledge-mode shuffled --evaluate-test
~~~

## 5. Reproduce the comparison models without redistributing their code

~~~powershell
python scripts/fetch_resources.py --upstream melbert dags roppt contrastwsd --accept-upstream-terms
python scripts/configure_upstream.py
~~~

Third-party implementations are cloned into ignored external/ directories at the commit ids recorded in resources/upstream_sources.json. The distributed repository contains our execution/protocol adapters, not copies of the authors' model architecture files. Compatibility edits are performed only on downloaded copies: AdamW import updates, DAGS missing utilities and input-path/target-field handling, a DAGS duplicate-pooling shape correction, and ContrastWSD tuple/truncation/TSV parsing corrections. DAGS utilities are sourced from the downloaded RoPPT repository. These adaptations are disclosed, not described as unchanged official runs.

| Model | Official repository | Strict local coverage |
|---|---|---|
| MelBERT | [jin530/MelBERT](https://github.com/jin530/MelBERT) | Five datasets; target end-to-end and separate frozen transfer |
| DAGS-Auto | [VILAN-Lab/Metaphor-DAGS](https://github.com/VILAN-Lab/Metaphor-DAGS) | Five datasets; automatic spaCy dependency inputs |
| RoPPT-Auto | [MajiBear000/RoPPT](https://github.com/MajiBear000/RoPPT) | Five datasets; same automatic dependency inputs |
| ContrastWSD-Aligned | [melzohbi/ContrastWSD](https://github.com/melzohbi/ContrastWSD) | VUA-18, standard VUA-Verb, VUA-20 only |

Build baseline inputs (the following commands preserve the baseline's released VUA-18 order and ids; normalized keys only verify instance alignment and do not rewrite the text fed to its model):

~~~powershell
python scripts/prepare_baselines.py --dataset vua18 --melbert-vua18 downloads/melbert_data/VUA18 --wsd-data downloads/contrastwsd_data --graphs
python scripts/prepare_baselines.py --dataset vua_verb_standard --wsd-data downloads/contrastwsd_data --graphs
python scripts/prepare_baselines.py --dataset vua20 --wsd-data downloads/contrastwsd_data --graphs
python scripts/prepare_baselines.py --dataset mohx --graphs
python scripts/prepare_baselines.py --dataset trofi --graphs
~~~

The original CLMR CSV and baseline TSV releases contain legacy non-ASCII encoding differences and different target orderings. These existing representations are preserved to match the earlier runs. Do not treat the underlying text files as byte-identical across methods. Labels, target positions, POS and normalized sentence identities are checked as multisets; source row ids/order are retained for WSD matching and graph preprocessing.

Train a comparison model using the same launcher and fixed seeds:

~~~powershell
python scripts/run_experiments.py --model melbert --dataset vua18
python scripts/run_experiments.py --model dags --dataset vua18
python scripts/run_experiments.py --model roppt --dataset vua18
python scripts/run_experiments.py --model contrastwsd --dataset vua18
~~~

Replace vua18 with vua_verb_standard, vua20, mohx or trofi as supported. Baseline-specific training settings follow the previous formal reruns; shared protocol does not mean identical model hyperparameters. MelBERT uses 3 epochs on VUA, 8 on MOH-X and 4 on TroFi; the other three adapters use 10 epochs. Each uses dev-only selection and one final test. ContrastWSD has no released MOH-X/TroFi contextual WSD inputs; the launcher rejects these combinations instead of inventing a comparable result.

MrBERT ([repository](https://github.com/cnunlp/MrBERT-Metaphor-Relation-BERT-)), MisNet ([repository](https://github.com/SilasTHU/MisNet)), CLCL ([repository](https://github.com/zhjjn/CLCL)), and MiceCL ([repository](https://github.com/Evilxya/MiceCL)) are literature comparisons only in this package. Their earlier resource/split limitations prevent a claim of fully aligned strict reruns. No one-command strict reproduction result is fabricated for them.

## 6. Frozen-encoder transfer is a different experiment

~~~powershell
python scripts/run_experiments.py --model clmr --dataset mohx --mode frozen
python scripts/run_experiments.py --model clmr --dataset trofi --mode frozen
python scripts/run_experiments.py --model melbert --dataset mohx --mode frozen
python scripts/run_experiments.py --model dags --dataset mohx --mode frozen
python scripts/run_experiments.py --model roppt --dataset mohx --mode frozen
~~~

Use trofi in the last three commands for the second target dataset. The launcher first trains source models on VUA-18, then trains target heads inside the same fixed folds with the encoders frozen.

The CLMR frozen-transfer source recipe uses hard-negative learning without the final main model's R-Drop. clmr_transfer_source.json and clmr_frozen_target.json preserve that recipe, and the launcher trains the required source models locally. MelBERT frozen-transfer source training also uses class weight 3 rather than the primary VUA-18 rerun's class weight 5. Report these protocols separately and disclose their source settings.

## 7. Outputs and analysis

Outputs are separated by mode/model/dataset/seed/fold under artifacts/runs/. summarize_runs.py pools outer folds and writes per-seed metrics plus mean/sample-SD to summary.json. It never averages fold F1 to form the main result.

~~~powershell
python scripts/summarize_runs.py --run-root artifacts/runs/train/clmr/mohx
~~~

The original statistical, ablation, subgroup and format-conversion utilities remain in clmr/scripts/ and scripts/processing/. Run analyses after training has produced the required predictions and metadata.

