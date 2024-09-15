from datasets import load_dataset
from datasets import concatenate_datasets

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import numpy as np
from rouge import Rouge
rouge = Rouge()
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import nltk

model_id = r"download/flan_t5_large"

dataset = load_dataset(
    "json",
    data_files={
        "train": "sum_output/dataset_t5/sum_t5_v1_train.jsonl",
        "test": "sum_output/dataset_t5/sum_t5_v1_test.jsonl",
    },
)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["input_text"], truncation=True),
    batched=True,
    remove_columns=["input_text", "output_text"],
)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["output_text"], truncation=True),
    batched=True,
    remove_columns=["input_text", "output_text"],
)
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")

def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    #inputs = ["question: " + item for item in sample["question"]]
    inputs = [item for item in sample["input_text"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=400, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    #labels = tokenizer(text_target=sample["output_text"], max_length=60, padding=padding, truncation=True)
    labels = tokenizer(text=sample["output_text"], max_length=60, padding=padding, truncation=True)
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "output_text"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

model = AutoModelForSeq2SeqLM.from_pretrained(r"download/flan_t5_large")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.eos_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
    for pred, label in zip(decoded_preds, decoded_labels):
        if len(pred) == 0 or len(label) == 0:
            result = {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
        else:
            scores = rouge.get_scores(label, pred)
            result = scores[0]
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))
    return {k: float(np.mean(v)) for k, v in score_dict.items()}


# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

training_args = Seq2SeqTrainingArguments(
    
    output_dir="x_retrieval/results/t5_res_xl",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    bf16=False,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    num_train_epochs=3,
    warmup_steps=100,
    # logging & evaluation strategies
    logging_dir=f"logs",
    logging_strategy="steps",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    # metric_for_best_model="overall_f1",
    # push to hub parameters
    report_to="tensorboard",
    push_to_hub=False,
    generation_max_length=60,
    
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"],
    compute_metrics=None,
)

trainer.train()
model.save_pretrained("x_retrieval/results/t5_res_xl/final_save")
print("model saved to x_retrieval/results/t5_res_xl/final_save")
