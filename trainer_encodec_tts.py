from datasets import load_dataset
from jiwer import wer
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments, BartForConditionalGeneration
)

# Load dataset and tokenizer
dataset = load_dataset("voidful/librispeech_encodec")
tokenizer = AutoTokenizer.from_pretrained("voidful/bart-base-unit")
model = BartForConditionalGeneration.from_pretrained("voidful/bart-base-unit")

# Split the dataset into training and validation sets

train_dataset = dataset['trainclean100']
valid_dataset = dataset['validationclean']

# Set training parameters
training_args = Seq2SeqTrainingArguments(
    output_dir="./training_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=100,
    predict_with_generate=True,
    fp16=True,
    gradient_accumulation_steps=2,
)

# Define a data collator to handle tokenization
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


def pad_sequences(sequences, max_length, padding_value):
    return [sequence + [padding_value] * (max_length - len(sequence)) for sequence in sequences]


# Define training and validation functions
def process_data_to_model_inputs(batch):
    input_ids = []
    attention_mask = []
    decoder_input_ids = []
    labels = []

    max_length = 1023  # You can set this to a suitable value based on your dataset

    for b in range(len(batch['text'])):
        # first layer AR data
        data = tokenizer(batch["text"][b], padding='max_length', truncation=True, max_length=max_length)
        encode_input = tokenizer.convert_tokens_to_ids([f"v_tok_{u}" for u in batch[f'encodec_{0}'][b]])
        decoder_input_id = [tokenizer.bos_token_id] + encode_input
        label = encode_input + [tokenizer.eos_token_id]
        input_ids.append(data['input_ids'])
        attention_mask.append(data['attention_mask'])
        decoder_input_ids.append(decoder_input_id)
        labels.append(label)

        # 1-7 layer NAR data
        for i in range(1, 8):
            decoder_input_id = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u + (i - 1) * 1000}" for u in batch[f'encodec_{i - 1}'][b]])
            label = tokenizer.convert_tokens_to_ids([f"v_tok_{u + i * 1000}" for u in batch[f'encodec_{i}'][b]])
            input_ids.append(data['input_ids'])
            attention_mask.append(data['attention_mask'])
            decoder_input_ids.append(decoder_input_id)
            labels.append(label)

    # Pad decoder_input_ids and labels
    decoder_input_ids = pad_sequences(decoder_input_ids, max_length=max_length, padding_value=tokenizer.pad_token_id)
    labels = pad_sequences(labels, max_length=max_length, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels
    }


def filter_examples(example):
    return len(example[f"encodec_0"]) <= 1000


train_dataset = train_dataset.filter(filter_examples)
valid_dataset = valid_dataset.filter(filter_examples)

train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    remove_columns=train_dataset.column_names,
    batched=True,
    batch_size=training_args.per_device_train_batch_size
)
valid_dataset = valid_dataset.map(process_data_to_model_inputs,
                                  remove_columns=valid_dataset.column_names,
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
)

# Start training
trainer.train()
# trainer.evaluate()
