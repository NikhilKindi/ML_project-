# ML_project-

This repository contains the complete codebase for a project on **symbolic music modeling** using **Transformer-based language models** and **LSTM-based recurrent neural networks**. The goal of the project is to study **scaling behavior**, **computational efficiency**, and **generation quality** as model size increases.

The project follows a full end-to-end pipeline:

> **MIDI → ABC conversion → data cleaning → tokenization → train/val/test split → Transformer training and LSTM training → scaling analysis → music generation**


## Important Note on Large Files (Please Read)

Due to GitHub file size limitations, the following large files are **not included** in this repository:

* `train.bin`, `val.bin`, `test.bin`, `abc_char_tokens`
* Cleaned ABC dataset files
* Model checkpoints (`.pt`, `.ckpt`)
* Large log files
* Token File 

These files exceed GitHub’s upload limits (25–100 MB per file).
**All results reported in the paper can be reproduced** using the scripts provided in this repository.

Instructions to regenerate all data and results are provided below.

## Dataset

### Lakh MIDI Dataset

We use the **Lakh MIDI Dataset (LMD)**, which contains over **176,000 MIDI files** spanning diverse musical genres.


## Step 1: MIDI to ABC Conversion

MIDI files are converted to **ABC notation**, a text-based symbolic music representation, using the `midi2abc` command-line tool.

### Command used:

```bash
midi2abc input.mid -o output.abc
```

This conversion produces a very large text corpus (billions of characters).


## Step 2: Data Cleaning

The raw ABC files contain noise and conversion artifacts. Cleaning steps include:

* Removing corrupted or non-musical files
* Removing comments and invalid symbols
* Standardizing required ABC headers (key, meter, note length)
* Discarding files with:

  * Fewer than **50 characters**
  * More than **200,000 characters**

Only files containing sufficient musical content are retained.

## Step 3: Tokenization

### Character-Level Tokenization

* Each character in ABC notation is treated as a token
* Vocabulary size: **99**
* No.of Tokens : 
* Preserves full symbolic information
* Simple and robust for large-scale modeling

### Output format

Tokenized data is stored in binary format for efficiency:

* test.bin
* val.bin
* train.bin

> These binary files are **not uploaded** due to size constraints.

---

## Step 4: Train / Validation / Test Split

* 98% training
* 1% validation
* 1% test

For scaling experiments:

* Each model is trained for **exactly one epoch**
* **200M tokens** are used per model


## Transformer Models

### Implementation

* Decoder-only GPT-style Transformers
* Built using **nanoGPT**
* Modified for:

  * ABC dataset loading
  * Model scaling
  * Logging and evaluation

### Model Sizes

* Tiny (~1M parameters)
* Small (~6M parameters)
* Medium (~21M parameters)
* Large (~75M parameters)
* XL (~177M parameters)

All Transformer models use:

* Same tokenization
* Same learning rate schedule
* Same batch size (measured in tokens)
* One epoch of training (for scaling comparison)

### Configuration Files

Transformer configurations are stored in:
* Tranformer_training File
* transformer-and-rnn-training

Transformer training scripts are located in the **Transformer training files**.

---

## RNN (LSTM) Models

### Implementation

* Character-level LSTM language models
* Same dataset, tokenization, and optimizer as Transformers
* context window for feasible training time

### Model Sizes

Four LSTM models were trained with parameter counts comparable to the Transformer models:

* Tiny
* Small
* Medium
* Large

LSTM architectures and hyperparameters are defined in the **RNN training files**.

---

## Step 5: Training and Logging

For each model, we record:

* Training loss
* Validation loss (after 1 epoch)
* Wall-clock time per epoch
* GPU memory usage
* Time per million parameters

Logs are saved locally and parsed using analysis scripts (not all logs are uploaded due to size).

---

## Step 6: Scaling Law Analysis

A power-law scaling relationship is fit:
[
L(N) = aN^{-\alpha} + c
]

Where:

* (N) is the number of parameters
* (L(N)) is validation loss after one epoch

Scaling plots are generated for:

* Transformers
* LSTMs
* Combined comparison (Transformer vs LSTM)

---

## Step 7: Music Generation

Generated 10 samples for each.

### Unconditional Generation

Models generate music from scratch. Larger Transformers produce:

* Longer coherent phrases
* More stable rhythm
* Clearer tonal structure

### Conditional Generation

Models generate continuations given a musical prompt.

The **XL Transformer**, trained on ~200M tokens, achieves the best quantitative and qualitative performance.

---

## Software Requirements

* Python 3.9+
* PyTorch
* NumPy
* Pandas
* SciPy
* Matplotlib
* tqdm
* joblib
* `midi2abc` (command-line tool)
* To convert the abc files to midi used SPUDS online ABC converter

GPU acceleration (CUDA) is strongly recommended.


---

## Notes

* Large binary files and checkpoints are excluded due to GitHub size limits
* All results in the report can be reproduced using the provided scripts
* The project emphasizes **fair, controlled scaling comparisons**

