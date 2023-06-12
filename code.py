import pandas as pd
import csv
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
from numpy import sqrt 

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        inputs = {k: inputs[k].squeeze() for k in inputs}

        label = torch.tensor(self.labels[idx]).float()

        return {**inputs, 'labels': label}

in_data = pd.read_csv('in.tsv', delimiter='\t', quoting=csv.QUOTE_NONE, names=['hex', 'text'], na_values='NaN', skiprows=1)
exp_data = pd.read_csv('expected.tsv', delimiter='\t', quoting=csv.QUOTE_NONE, names=['year'], na_values='NaN', skiprows=1)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-multilingual-cased')

scaler = StandardScaler()
scaled_labels = scaler.fit_transform(exp_data['year'].values.reshape(-1, 1))

dataset = TextDataset(in_data['text'].values, scaled_labels, tokenizer)

train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=1)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

criterion = torch.nn.MSELoss()

epochs = 10

accumulation_steps = 4

for epoch in range(epochs):
    model.train()  # ustawiamy model w tryb treningowy
    total_loss = 0
    for i, batch in enumerate(tqdm(train_dataloader)):
        # Przenosimy dane na odpowiednie urządzenie (GPU lub CPU)
        batch = {k: v for k, v in batch.items()}

        outputs = model(**batch)

       
        loss = criterion(outputs.logits.squeeze(), batch["labels"])

        loss.backward()

        if (i+1) % accumulation_steps == 0:  # Aktualizacja wag co 4 mini-batche
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        torch.cuda.empty_cache()
        
    print(f"Train loss in epoch {epoch+1}: {total_loss/len(train_dataloader)}")
