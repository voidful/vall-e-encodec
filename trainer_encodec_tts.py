from datasets import load_dataset
from jiwer import wer
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments, DataCollatorWithPadding
)

# Load dataset and tokenizer
from encodec_bart_model import BartEncodecForConditionalGeneration

dataset = load_dataset("voidful/librispeech_encodec")
tokenizer = AutoTokenizer.from_pretrained("voidful/bart-base-unit")
model = BartEncodecForConditionalGeneration.from_pretrained("voidful/bart-base-unit")

# Split the dataset into training and validation sets
train_dataset = dataset['trainclean100']
valid_dataset = dataset['validationclean']

# Set training parameters
training_args = Seq2SeqTrainingArguments(
    output_dir="./training_output",
    num_train_epochs=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=10,
    predict_with_generate=True,
    learning_rate=5e-4,
    fp16=True,
    gradient_accumulation_steps=2
)
data_collator = DataCollatorWithPadding(tokenizer)


# Define training and validation functions
def process_data_to_model_inputs(batch):
    input_ids = []
    attention_mask = []
    decoder_input_ids = []
    labels = []

    max_length = 1023  # You can set this to a suitable value based on your dataset

    for b in range(len(batch['text'])):
        data = tokenizer(batch["text"][b], padding='max_length', truncation=True, max_length=max_length)
        input_ids.append(data['input_ids'])
        attention_mask.append(data['attention_mask'])

        # first layer AR data
        encode_input = tokenizer.convert_tokens_to_ids([f"v_tok_{u}" for u in batch[f'encodec_{0}'][b]])
        decoder_input_id = [model.config.decoder_start_token_id] + encode_input
        label = encode_input + [tokenizer.eos_token_id]
        decoder_input_ids.append(decoder_input_id)
        labels.append(label)

        # 1-7 layer NAR data
        for i in range(1, 8):
            decoder_input_id = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u + (i - 1) * 1024}" for u in batch[f'encodec_{i - 1}'][b]])
            label = tokenizer.convert_tokens_to_ids([f"v_tok_{u + i * 1024}" for u in batch[f'encodec_{i}'][b]])
            input_ids.append(data['input_ids'])
            attention_mask.append(data['attention_mask'])
            decoder_input_ids.append(decoder_input_id)
            labels.append(label)

    def pad_sequences_and_create_masks(sequences, max_length, padding_value):
        padded_sequences = [sequence + [padding_value] * (max_length - len(sequence)) for sequence in sequences]
        attention_masks = [[1 if token != padding_value else 0 for token in sequence] for sequence in padded_sequences]
        return padded_sequences, attention_masks

    # Pad decoder_input_ids and labels
    decoder_input_ids, decoder_attention_mask = pad_sequences_and_create_masks(decoder_input_ids, max_length=max_length,
                                                                               padding_value=tokenizer.pad_token_id)
    labels, _ = pad_sequences_and_create_masks(labels, max_length=max_length, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        'decoder_attention_mask': decoder_attention_mask,
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
train_dataset = train_dataset.shuffle(seed=42)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = [i[i != -100] for i in labels]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute WER
    wer_value = wer([" ".join(filter(None, i.split("v_tok_"))) for i in decoded_labels],
                    [" ".join(filter(None, i.split("v_tok_"))) for i in decoded_preds])
    print("pred_result")
    print("=================================")
    for i in range(10):
        print("target:" + labels[i])
        print("pred:" + predictions[i])
        print("-----------------")
    print("=================================")
    return {"wer": wer_value}


# Create the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()
# trainer.evaluate()
