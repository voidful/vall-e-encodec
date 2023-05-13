import torch
from IPython.display import Audio
from datasets import load_dataset
from encodec import EncodecModel
from transformers import AutoTokenizer

from encodec_bart_model import BartEncodecForConditionalGeneration

dataset = load_dataset("voidful/librispeech_encodec", split="trainclean100")
dataset = dataset.shuffle(seed=42).select(range(30))
tokenizer = AutoTokenizer.from_pretrained("voidful/bart-base-unit")
model = BartEncodecForConditionalGeneration.from_pretrained("./checkpoint-45356/")
model = model.to('cuda')

inputs = tokenizer(dataset["text"][0],
                   padding="max_length",
                   truncation=True,
                   max_length=1024,
                   return_tensors="pt").to('cuda')
decode_ar = model.generate(**inputs, max_length=1024, num_beams=1, do_sample=False)
decoded_tok = tokenizer.batch_decode(decode_ar, skip_special_tokens=True)[0]


def nar_decode(batch_code, layer=0):
    base_input = inputs
    base_input['decoder_input_ids'] = batch_code
    decode_nar = model.forward(**base_input).logits

    id_range_start, id_range_end = tokenizer.convert_tokens_to_ids(
        f'v_tok_{0 + 1024 * layer}'), tokenizer.convert_tokens_to_ids(f'v_tok_{1024 + 1024 * layer}')

    # Create a tensor where values are equal to their own indices
    indices = torch.arange(decode_nar.size(-1)).to(decode_nar.device)

    # Create a mask for the range
    mask = (indices >= id_range_start) & (indices < id_range_end)

    # Set values out of range to very low value
    decode_nar_masked = torch.where(mask, decode_nar, torch.tensor(float('-inf')).to(decode_nar.device))

    # Get the argmax within the range
    return torch.argmax(decode_nar_masked, dim=-1)


# all ground truth unit
layer_list = []
for layer_i in range(8):
    encode_input = tokenizer("".join([f"v_tok_{u + layer_i * 1024}" for u in dataset[f'encodec_{layer_i}'][0]]),
                             return_tensors='pt', add_special_tokens=False).to('cuda')
    encode_input = encode_input['input_ids']
    layer_list.append(encode_input)

# using model prediction

layer_list = []

# use ar prediction
layer_list.append(decode_ar[:, 1:-1])

# use ground truth ar prediction
layer_i = 0
encode_input = tokenizer("".join([f"v_tok_{u + layer_i * 1024}" for u in dataset[f'encodec_{layer_i}'][0]]),
                         return_tensors='pt', add_special_tokens=False).to('cuda')
encode_input = encode_input['input_ids']
layer_list.append(encode_input)

# iterative predict nar code
for layer in range(1, 8):
    layer_list.append(nar_decode(layer_list[-1], layer))

# covert ar+nar code into encodec code
encodec_code = []
for layer, layer_ids in enumerate(tokenizer.batch_decode(torch.cat(layer_list))):
    layer_ids = layer_ids.replace("</s>", "")
    encodec_code.append([int(i) - layer * 1024 for i in layer_ids.split('v_tok_') if len(i) > 0])

encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(6.0)

# synthesize audio
Audio(encodec_model.decode([(torch.tensor(encodec_code).unsqueeze(0).to('cpu'), None)]).detach().numpy()[0], rate=24000)
