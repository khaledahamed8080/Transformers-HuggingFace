from transformers import pipeline, AutoTokenizer,AutoModelForSequenceClassification

classifier=pipeline("sentiment-analysis")
output=classifier("i am feeling fulfilled from botom of my heart")
print(output)

model_name="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForSequenceClassification.from_pretrained(model_name)
classy=pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)
opt=classy("i am feeling fulfilled from botom of my heart")

print(opt)

seq="Using a Transformer Network is simple"
res=tokenizer(seq)
print(res)
tokens=tokenizer.tokenize(seq)
print(tokens)
ids=tokenizer.convert_tokens_to_ids(tokens)
print(ids)
decode_string=tokenizer.decode(ids)
print(decode_string)