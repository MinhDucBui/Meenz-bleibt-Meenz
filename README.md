<h1 align="center">Meenz Bleibt Meenz</h1>
<p align="center"><em>But Large Language Models Do Not Speak the Dialect of Mainz</em></p>



This repository contains code, data, and evaluation scripts for our LREC 2026 paper:

[![Paper Link (soon)](https://img.shields.io/badge/Paper-LREC--COLING%202026-blue)](link-to-paper)



> **Abstract:** We present the first NLP research on Meenzerisch and introduce a digital dictionary containing 2,351 dialect words with Standard German definitions. Our experiments show that state-of-the-art LLMs struggle dramatically: the best model achieves only 6.27% accuracy on definition generation and 1.51% on word generation, highlighting the urgent need for additional resources and research on German dialects.

---

## 📊 Dataset: Mainz Dialect Dictionary

### Overview

The **Mainz Dialect Dataset** is derived from Karl Schramm's 1966 "Mainzer Wörterbuch" through a semi-automatic digitization pipeline.

### Examples

| Meenzerisch Word | Standard German Definition | English Translation |
|------------------|---------------------------|---------------------|
| Aaweiderworschd | Salzgurke | pickled cucumber |
| Bitzelwasser | Mineralwasser mit Kohlensäuregehalt | carbonated mineral water |
| Schimmes | Hunger | hunger |

### Access to Dataset

The dataset is distributed under **CC BY-NC-ND 4.0**.

**To request access:** [Instructions will be provided soon]

---

## 🚀 Installation & Setup
```bash
# Clone the repository
git clone https://github.com/MinhDucBui/Meenz-bleibt-Meenz
cd Meenz-bleibt-Meenz

# Install dependencies
pip install -r requirements.txt
```

⚠️ IMPORTANT

- You will receive the train/dev/test splits as `.csv` files.  
  Please place them in `output/extractor/`.

- Insert the correct model paths in the `sh/` scripts before running the models.


---


### 📝 Evaluation: LLM-as-a-Judge

Automatically evaluate whether generated and gold definitions are semantically equivalent via LLMaaJ.
```bash
python src/inference.py \
    --input_files "${files[@]}" \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --mode "evaluator" \
    --batch_size 64 \
    --max_new_tokens 8
```

## 🧪 Experiments



### Experiment 1: Definition Generation

**Task:** Generate definitions for Meenzerisch words (in Standard German).

**Run the experiment:**
```bash
# Main experiment
sh sh/generator_instruct_mainz.sh

# Or run directly:
python src/inference.py \
    --model_name "/path/to/model" \
    --mode "generator" \
    --batch_size 128 \
    --max_new_tokens 512
```

**English baseline comparison:**
```bash
sh sh/generator_instruct_english.sh
```

---

### Experiment 2: Dialect Word Generation

**Task:** Generate Meenzerisch words from their Standard German definitions.

**Run the experiment:**
```bash
# Main experiment
sh sh/generatorreverse_mainz.sh

# Or run directly:
python src/inference.py \
    --model_name "/path/to/model" \
    --mode "generatereversed" \
    --batch_size 64 \
    --max_new_tokens 128
```

**English baseline comparison:**
```bash
sh sh/generatorreverse_english.sh
```

---

### Additional Experiments

#### Few-Shot Learning (Section 7.1)

Test whether providing in-context examples improves performance.

```bash
# For definition generation (random selection)
python src/inference.py \
    --model_name "/path/to/model" \
    --mode "generator_fewshot_random" \
    --shots "$shots" \
    --batch_size 64 \
    --max_new_tokens 128

# For definition generation (edit distance selection)
python src/inference.py \
    --model_name "/path/to/model" \
    --mode "generator_fewshot_editdistance" \
    --shots "$shots" \
    --batch_size 64 \
    --max_new_tokens 128

# For word generation (random selection)
python src/inference.py \
    --model_name "/path/to/model" \
    --mode "generatereversed_fewshot_random" \
    --shots "$shots" \
    --batch_size 32 \
    --max_new_tokens 512
```

#### Automatic Rule Extraction (Section 7.2)

Inject automatically extracted dialect-to-Standard German mapping rules.

```bash
# Definition Generation
python src/inference.py \
    --model_name "/path/to/model" \
    --mode "generator_rulebased" \
    --batch_size 32 \
    --max_new_tokens 512

# Word Generation
python src/inference.py \
    --model_name "/path/to/model" \
    --mode "generatereversed_rulebased" \
    --batch_size 64 \
    --max_new_tokens 512
```

---

## 📄 Citation

If you use this dataset or code, please cite:

Soon

