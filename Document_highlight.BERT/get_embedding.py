import torch
from transformers import *

model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-uncased'

if __name__ == '__main__':
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights).to('cuda')
    tokenized_text = tokenizer.tokenize("Here is some text to encode")
    print (tokenized_text)
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)]).to('cuda')
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
    print (last_hidden_states)
