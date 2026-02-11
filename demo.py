from transformers import pipeline, AutoTokenizer,AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
classifier=pipeline("sentiment-analysis")
output=classifier("i am feeling fulfilled from botom of my heart")
print(output)

model_name="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForSequenceClassification.from_pretrained(model_name)
classy=pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)
opt=classy("i am feeling fulfilled from botom of my heart")

print(opt)

text=["I love animals and i hve been taking care of them since my childhood","You are beautiful"]

out=classy(text)
print(out)
batch=tokenizer(text, padding=True,truncation=True,max_length=512,return_tensors="pt")
print(batch)

with torch.no_grad():
    outpt=model(**batch)
    print(outpt)
    predictions=F.softmax(outpt.logits,dim=1)
    print(predictions)
    labels=torch.argmax(predictions,dim=1)
    print(labels)



