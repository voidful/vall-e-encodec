from datasets import load_dataset
from jiwer import wer
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from encodec_bart_model import BartEncodecForConditionalGeneration

# Load dataset and tokenizer
dataset = load_dataset("voidful/librispeech_encodec")
tokenizer = AutoTokenizer.from_pretrained("voidful/bart-base-unit")
model = BartEncodecForConditionalGeneration.from_pretrained("voidful/bart-base-unit")

# Split the dataset into training and validation sets

train_dataset = dataset['trainclean100']
# train_dataset = dataset['validationclean']
valid_dataset = dataset['validationclean']

# Set training parameters

training_args = Seq2SeqTrainingArguments(
    output_dir="./training_output",
    logging_dir="./logs",
    evaluation_strategy="epoch",
    num_train_epochs=50,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=0,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    learning_rate=5e-5,
    weight_decay=0,
    predict_with_generate=True,
    generation_max_length=300,
    fp16=True,
    save_total_limit=3,
)

# Define a data collator to handle tokenization
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# Define training and validation functions
def process_data_to_model_inputs(batch):
    max_len = 1023
    labels = tokenizer(batch["text"], padding=True, truncation=True, max_length=max_len).input_ids
    # Replace pad_token_id (0) with -100
    labels = [[-100 if token_id == tokenizer.pad_token_id else token_id for token_id in seq] for seq in labels]
    batch["labels"] = labels

    input_datas = []
    for b in range(len(batch['text'])):
        encodec_input = []
        for i in range(8):
            encodec_input.append(
                tokenizer.convert_tokens_to_ids([f"v_tok_{u + i * 1024}" for u in batch[f'encodec_{i}'][b]]))
        input_datas.append(encodec_input)
    # Pad the input data sequences and create attention masks
    padded_input_datas = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(data).transpose(0, 1) for data in input_datas], 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id)  # pad 到最長
    # [BSZ, 8, MAX_SEQ] 切到 max_len
    padded_input_datas = padded_input_datas[:, :max_len].transpose(1, 2)  
    attention_masks = torch.zeros((len(input_datas), max_len), dtype=int)  # 預設是 0
    for data_id, data in enumerate(input_datas):
        first_of_eight_code = data[0]
        attention_masks[data_id, :len(first_of_eight_code)] = 1  # 有資料的是 1
    batch["input_ids"] = padded_input_datas    # torch.Size([BSZ, 8, SEQ_LEN])
    batch["attention_mask"] = attention_masks  # torch.Size([BSZ, SEQ_LEN])
    return batch


def filter_examples(example):
    return len(example[f"encodec_0"]) <= 1000


train_dataset = train_dataset.filter(filter_examples)
valid_dataset = valid_dataset.filter(filter_examples)

train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=training_args.per_device_train_batch_size
)
valid_dataset = valid_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=training_args.per_device_eval_batch_size
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = [i[i != -100] for i in labels]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute WER
    wer_value = wer(decoded_labels, decoded_preds)
    print("pred_result")
    print("=================================")
    for i in range(10):
        print(decoded_labels[i], " ///// ", decoded_preds[i])
    print("=================================")
    return {"wer": wer_value}


# Create the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
# trainer.evaluate()
