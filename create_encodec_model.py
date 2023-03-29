import copy

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
pretrained = AutoModel.from_pretrained("google/long-t5-tglobal-base")

print(tokenizer.tokenize("Hello world v_tok_10v_tok_1"))
add_tokens = [f"v_tok_{u}" for u in range(1024 * 9)]
origin_vocab_size = tokenizer.vocab_size
print("===ADD TOKEN===")
num_added_toks = tokenizer.add_tokens(add_tokens)
print('We have added', num_added_toks, 'tokens')
print(origin_vocab_size, num_added_toks, len(tokenizer))
print(tokenizer.tokenize("Hello world v_tok_10v_tok_1"))
print("===============")
# reshape pretraining embedding
pretrained.resize_token_embeddings(origin_vocab_size + num_added_toks)
input_embedding = pretrained.get_input_embeddings()
state_dict_weight = input_embedding.state_dict()['weight']
print(state_dict_weight.shape, state_dict_weight[10:100].shape)
state_dict_weight[origin_vocab_size:origin_vocab_size + num_added_toks] = copy.copy(
    state_dict_weight[100:100 + num_added_toks])
pretrained.set_input_embeddings(input_embedding)
print("===============")

tokenizer.push_to_hub("voidful/long-t5-encodec-tglobal-base")
pretrained.push_to_hub("voidful/long-t5-encodec-tglobal-base")
