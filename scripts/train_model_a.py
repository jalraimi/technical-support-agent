from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_A = "distilbert-base-uncased"
LABELS = [
    "authentication", "network", "deployment", "database",
    "gpu", "api", "package", "general",
]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A)
model_a = AutoModelForSequenceClassification.from_pretrained(
    MODEL_A,
    num_labels=len(LABELS),
    label2id=label2id,
    id2label=id2label,
)

def tokenize_a(batch):
    return tokenizer_a(batch["text"], truncation=True, max_length=128)

def compute_cls_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
    }

# TODO(student 1): load/construct at least 12 examples per intent,
# then create stratified train/validation/test splits.

# 1. Load data
df = pd.read_csv("data/intents.csv")

# 2. Split 1: 70% Train, 30% Temporary (for val + test)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

# 3. Split 2: Divide the 30% temp in half -> 15% Validation, 15% Test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

def prepare_dataset(split_df):
    ds = Dataset.from_pandas(split_df.reset_index(drop=True))
    # Map text labels ("network", "gpu") to numerical IDs
    ds = ds.map(lambda ex: {"label": label2id[ex["label"]]})
    # Tokenize text using the function already defined in your script
    ds = ds.map(tokenize_a, batched=True)
    return ds

# 4. Convert DataFrames to Hugging Face Datasets with labels & tokenization
train_a = prepare_dataset(train_df)
val_a = prepare_dataset(val_df)
test_a = prepare_dataset(test_df)



args_a = TrainingArguments(
    output_dir="models/intent_classifier",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    report_to="none",
)

trainer_a = Trainer(
    model=model_a,
    args=args_a,
    train_dataset=train_a,
    eval_dataset=val_a,
    compute_metrics=compute_cls_metrics,
)
trainer_a.train()
trainer_a.evaluate(test_a)
trainer_a.save_model("models/intent_classifier")
tokenizer_a.save_pretrained("models/intent_classifier")
