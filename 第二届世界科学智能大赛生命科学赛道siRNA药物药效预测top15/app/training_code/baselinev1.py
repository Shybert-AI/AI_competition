import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter
from rich import print
from sklearn.metrics import precision_score, recall_score, mean_absolute_error
import pickle
from easydict import EasyDict
import math
import torch
from math import cos, pi

class GenomicTokenizer:
    def __init__(self, ngram=5, stride=2):
        self.ngram = ngram
        self.stride = stride
        
    def tokenize(self, t):
        try:
            t = t.upper()
            if self.ngram == 1:
                toks = list(t)
            else:
                toks = [t[i:i+self.ngram] for i in range(0, len(t), self.stride) if len(t[i:i+self.ngram]) == self.ngram]
            if len(toks[-1]) < self.ngram:
                toks = toks[:-1]
        except:
            pass

        return toks


class GenomicVocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {v:k for k,v in enumerate(self.itos)}
        
    @classmethod
    def create(cls, tokens, max_vocab, min_freq):
        freq = Counter(tokens)
        itos = ['<pad>'] + [o for o,c in freq.most_common(max_vocab-1) if c >= min_freq]
        return cls(itos)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SiRNADataset(Dataset):
    def __init__(self, df, columns, vocab, tokenizer, max_len,model_sign=None):
        self.df = df
        self.columns = columns
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_sign = model_sign

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        seqs = [self.tokenize_and_encode(row[col]) for col in self.columns]
        if self.model_sign is  None:
            target = torch.tensor(row['mRNA_remaining_pct'], dtype=torch.float)
        else:
            target = 0

        return seqs, target

    def tokenize_and_encode(self, seq):
        seq = str(seq)
        if ' ' in seq:  # Modified sequence
            tokens = seq.split()
        else:  # Regular sequence
            tokens = self.tokenizer.tokenize(seq)
        
        encoded = [self.vocab.stoi.get(token, 0) for token in tokens]  # Use 0 (pad) for unknown tokens
        padded = encoded + [0] * (self.max_len - len(encoded))
        return torch.tensor(padded[:self.max_len], dtype=torch.long)


class SiRNAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, hidden_dim=512, n_layers=2, dropout=0.1,model="gru"):
        super(SiRNAModel, self).__init__()
        self.model = model
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = PositionalEmbedding(d_model=200)
        if self.model == "gru":

            self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True, dropout=dropout)
            #self.gru = nn.LSTM(embed_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True, dropout=dropout)
            self.fc = nn.Sequential(nn.Linear(16384, 2560),
            nn.BatchNorm1d(2560),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2560, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1))
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #embedded = [self.embedding(seq) + self.position_embedding(seq) for seq in x]
        embedded = [self.embedding(seq) for seq in x]
        if self.model == "gru":
            outputs = []
            for embed in embedded:
                x, _ = self.gru(embed)
                x = self.dropout(x[:, -1, :])  # Use last hidden state
                outputs.append(x)

            # x_T,_ = self.gru(torch.cat(embedded, dim=1))
            # x_T = self.dropout(x_T[:, -1, :])
            # outputs.append(x_T)
            x = torch.cat(outputs, dim=1)
            x = self.fc(x)
        return x.squeeze()


def calculate_metrics(y_true, y_pred, threshold=30):
    mae = np.mean(np.abs(y_true - y_pred))

    y_true_binary = (y_true < threshold).astype(int)
    y_pred_binary = (y_pred < threshold).astype(int)

    mask = (y_pred >= 0) & (y_pred <= threshold)
    range_mae = mean_absolute_error(y_true[mask], y_pred[mask]) if mask.sum() > 0 else 100

    precision = precision_score(y_true_binary, y_pred_binary, average='binary')
    recall = recall_score(y_true_binary, y_pred_binary, average='binary')
    f1 = 2 * precision * recall / (precision + recall)
    score = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5
    return score


def train_model(model, train_loader, val_loader, criterion, optimizer,num_epochs=50, device='cuda'):
    model.to(device)
    best_score = -float('inf')
    best_model = None
    train_loss_list = []
    val_loss_list = []
    train_score_list = []
    val_score_list = []



    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        train_preds = []
        train_targets = []

        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = [x.to(device) for x in inputs]
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(targets.detach().cpu().numpy())

        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = [x.to(device) for x in inputs]
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        score = calculate_metrics(train_targets, train_preds)
        train_score_list.append(score)
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        score = calculate_metrics(val_targets, val_preds)
        val_score_list.append(score)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Validation Score: {score:.4f}')

        if score > best_score:
            best_score = score
            best_model = model.state_dict().copy()
            print(f'New best model found with socre: {best_score:.4f}')
            torch.save(model.state_dict(),f"model_{epoch}.pt")


    from matplotlib import pyplot as plt
    plt.plot(train_loss_list[2:],label="train_loss")
    plt.plot(val_loss_list[2:], label="val_loss")
    plt.legend()
    plt.savefig("gru_loss.png")

    return best_model

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = [x.to(device) for x in inputs]
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(target.numpy())

    y_pred = np.array(predictions)
    y_test = np.array(targets)
    
    score = calculate_metrics(y_test, y_pred)
    print(f"Test Score: {score:.4f}")


def warmup_cosine(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup_epoch=10):
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr = lr_min + (lr_max - lr_min) * (
                    1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# 将所有文本特征组合成一个字符串
def combine_features(row):
    return ' '.join([str(row['cell_line_donor']), str(row['siRNA_concentration']), str(row['Transfection_method']), str(row['modified_siRNA_sense_seq']),str(row['modified_siRNA_antisense_seq'])])
def calculate_gc(row):
    num = row.count("G") + row.count("C")
    return num/len(row)

def criterion_func(pre, target):
    loss1 = nn.MSELoss()
    loss2 = nn.SmoothL1Loss()
    loss3 = nn.HuberLoss()
    return loss1(pre,target) + loss2(pre,target) + loss3(pre,target)


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fix_seed = 3047
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    # Load data
    train_data = pd.read_csv(r'..\data\anl_train_data.csv')

    train_data['combined_features'] = train_data.apply(combine_features, axis=1)
    train_data['combined_features'] = train_data.apply(combine_features, axis=1)

    train_data['siRNA_sense_seq_gc'] = train_data["siRNA_sense_seq"].apply(calculate_gc)
    train_data['siRNA_antisense_seq_gc'] = train_data["siRNA_antisense_seq"].apply(calculate_gc)

    #columns = ['siRNA_antisense_seq', 'modified_siRNA_antisense_seq_list']
    columns = ['publication_id','gene_target_symbol_name','gene_target_ncbi_id','gene_target_species','siRNA_duplex_id','siRNA_sense_seq', 'siRNA_antisense_seq', 'cell_line_donor',
               'Transfection_method', 'modified_siRNA_sense_seq','modified_siRNA_antisense_seq',
               'modified_siRNA_sense_seq_list', ]#'modified_siRNA_antisense_seq_list']

   # 0.8303
    columns =  [
        #'id',
        #'publication_id',
        'gene_target_symbol_name',
        'gene_target_ncbi_id',
        'gene_target_species',
        'siRNA_duplex_id',
        'siRNA_sense_seq',
        'siRNA_antisense_seq',
        'cell_line_donor',
        'siRNA_concentration',
        #'concentration_unit',
        'Transfection_method',
        'Duration_after_transfection_h',
        'modified_siRNA_sense_seq',
        'modified_siRNA_antisense_seq',
        'modified_siRNA_sense_seq_list',
        'modified_siRNA_antisense_seq_list',
        "siRNA_sense_seq_gc",
        "siRNA_antisense_seq_gc"
        #'gene_target_seq',
        #'mRNA_remaining_pct'
    ]

    train_data.dropna(subset=columns + ['mRNA_remaining_pct'], inplace=True)
    train_data_split, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # Create vocabulary
    tokenizer = GenomicTokenizer(ngram=3, stride=3)

    all_tokens = []
    for col in columns:
        print(col)
        for seq in train_data[col]:
            if isinstance(seq,float):
                seq = str(seq)
            if ' ' in seq:  # Modified sequence
                all_tokens.extend(seq.split())
            else:
                all_tokens.extend(tokenizer.tokenize(seq))
    # 保存all_tokens，用于训练集的加载
    with open("train_all_tokens.pkl","wb") as f:
        pickle.dump(all_tokens,f)
    vocab = GenomicVocab.create(all_tokens, max_vocab=10000, min_freq=1)

    # Find max sequence length
    max_len = max(max(len(seq.split()) if ' ' in str(seq) else len(tokenizer.tokenize(str(seq)))
                      for seq in train_data[col]) for col in columns)
    #max_len = 96 if max_len < 96 else 96
    # Create datasets
    train_dataset = SiRNADataset(train_data, columns, vocab, tokenizer, max_len)
    val_dataset = SiRNADataset(val_data, columns, vocab, tokenizer, max_len)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize model
    model = SiRNAModel(len(vocab.itos))
    #state_dict = torch.load("model_199.pt")
    #model.load_state_dict(state_dict)
    #criterion = nn.MSELoss()
    criterion = criterion_func

    optimizer = optim.AdamW(model.parameters(),lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), weight_decay=0.001)

    train_model(model, train_loader, val_loader, criterion, optimizer,200, device)
