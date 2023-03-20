from transformers import AutoTokenizer

from model import BartEncodecForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("voidful/bart_base_encodec")
model = BartEncodecForConditionalGeneration.from_pretrained("voidful/bart_base_encodec")
encodec_input = tokenizer("hello world", return_tensors="pt").input_ids.unsqueeze(1).repeat(1, 7, 1)
print(model(input_ids=encodec_input,
            attention_mask=tokenizer("hello world", return_tensors="pt").attention_mask,
            labels=tokenizer("hello world", return_tensors="pt").input_ids).loss)
