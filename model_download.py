from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/flan-t5-base"  # Or any other HF model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained("./models/{}".format(model_name))
tokenizer.save_pretrained("./models/{}".format(model_name))
