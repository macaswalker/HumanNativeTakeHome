# Take-Home Assignment

We want to build a service that allows licensees to report data the expect is
in violation of local laws or regulations. 

The service will allow consumers to provide structured information about
- WHERE: where in the original piece of media
- WHY: what it is violation of
- HOW: how it is in violation of local laws and regulation

Key Assumptions:
- Trying to detail 'WHERE' means giving a token/character level description of where PII is in the text.
- Trying to detail 'WHY' means reporting back what kind of PII it is (name, email, etc)
- We will only focus on textual data
- We will only focus on a subsection of all PII: NAME, EMAIL, LOCATION, PHONE, URL.
- Synthetic data reflects real-world cases: we have trained and tested on synthetic data - hence, implicitly, we assume our synthetic data contains enough breadth to properly model real-world data.
- Cost constraints limit Third-Party API's: Due to scale of our data size, it would be prohibitve to use cloud API PII detection services.
- Only English supported (but framework flexible enough to include other languages)
- No persistence required

Please see ```Pii_Detection_walkthrough.ipynb``` for a full walkthrough of the model, how it is used and the discussion around Data and Datasets.


The structure of the repo is as follows:

```

HumanNativeTakeHome/
├── data/                          # Synthetic and test datasets
│   ├── Data.csv                  # Main training data (csv)
│   ├── Data.json                 # Main training data (json)
│   └── test/
│       ├── test_Data.csv         # Data created explicityly for test purporses (csv)
│       └── test_Data.json        # Data created explicityly for test purporses (json)
│
├── pii_ner_model/                # Trained DistilBERT PII token classification model
│   └── (Hugging Face files)      # tokenizer, config, weights
│
├── scripts/                      # CLI scripts for common tasks
│   ├── generate_dataset.py       # Generate main training dataset
│   ├── generate_test_dataset.py  # Generate test dataset
│   └── train_model.py            # Train BERT model on token-labeled PII
│
├── src/
│   ├── data/
│   │   ├── data_generation.py     # Faker-based blog (6 sentences) + PII generator
│   │   └── dataset_builder.py     # Wraps generation into a saved dataset (csv and json)
│   │
│   ├── models/
│   │   └── utils.py               # Tag mappings: tag2id, id2tag
│   │
│   ├── training/
│   │   ├── train.py               # Core BERT training loop
│   │   ├── dataset.py             # PyTorch Dataset class for token-level labeling
│   │   └── token_classifier.py    # Token-based inference for raw text
│   │
│   ├── evaluation/
│   │   └── token_classifier.py    # Evaluate model on dataframe
│   │
│   ├── presidio/
│   │   ├── detector.py            # Simple wrapper over default Presidio detection
│   │   └── anonymizer.py          # Redact PII using detection results
│
├── Pii_Detection_walkthrough.ipynb  # Full end-to-end project walkthrough
│
├── .gitattributes                # Git LFS configuration for model weights
├── requirements.txt              # Python dependencies
└── README.md                     # Project explanation, run through, asssumptions

```

To run through the scripts yourself, please follow these instructions:

```
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data
python scripts/generate_dataset.py

# 3. Generate test data
python scripts/generate_test_dataset.py

# 4. Train the DistilBERT token classification model
python scripts/train_model.py

# Evaluation code is in the notebook

```

