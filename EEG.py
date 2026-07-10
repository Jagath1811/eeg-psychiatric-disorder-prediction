# ==========================================================
# 4-CLASS ARCHITECTURE SWAP TEST
# Replace MLP with Hybrid Deep Model
# Keep: ClassCDI + COH-only + top_k=200
# No ensemble, no focal loss
# ==========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# LOAD FULL DATA (945 samples)
# ==========================================================

df = pd.read_csv("EEG_machinelearing_data_BRMH.csv")

group_map = {
    "Healthy control": 0,
    "Mood disorder": 1,
    "Anxiety disorder": 1,
    "Schizophrenia": 2,
    "Addictive disorder": 3,
    "Trauma and stress related disorder": 3,
    "Obsessive compulsive disorder": 3
}

df = df[df["main.disorder"].isin(group_map.keys())]
df["label"] = df["main.disorder"].map(group_map)

drop_cols = ["no.", "sex", "age", "eeg.date",
             "education", "IQ",
             "main.disorder", "specific.disorder"]

X_df = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
feature_names = np.array(X_df.columns.tolist())

X = X_df.values
y = df["label"].values

print("Total Samples:", len(y))
print("Number of Classes:", len(np.unique(y)))

bands = ["delta","theta","alpha","beta","gamma"]

# ==========================================================
# PREPROCESS
# ==========================================================

def preprocess(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

# ==========================================================
# CLASS-SPECIFIC CDI
# ==========================================================

def class_specific_cdi(X_train, y_train):
    classes = np.unique(y_train)
    scores = np.zeros(X_train.shape[1])

    for c in classes:
        y_bin = (y_train == c).astype(int)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_bin)
        imp = np.abs(model.coef_[0])
        scores += imp

    return scores

# ==========================================================
# COH FILTER
# ==========================================================

def filter_coh(idx):
    selected = feature_names[idx]
    mask = ["COH." in f for f in selected]
    return idx[np.array(mask)]

# ==========================================================
# HYBRID MODEL (Multi-class version)
# ==========================================================

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim,1)

    def forward(self,x):
        weights = torch.softmax(self.attn(x),dim=1)
        return torch.sum(weights*x,dim=1)

class HybridModel(nn.Module):
    def __init__(self,input_dim,num_classes):
        super().__init__()

        self.conv = nn.Conv1d(input_dim,input_dim,
                              kernel_size=3,padding=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,
            dim_feedforward=128,
            dropout=0.4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,num_layers=1
        )

        self.bilstm = nn.LSTM(
            input_dim,64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attn = AttentionPooling(128)

        self.fc = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,num_classes)
        )

    def forward(self,x):
        x = x.transpose(1,2)
        x = torch.relu(self.conv(x))
        x = x.transpose(1,2)
        x = self.transformer(x)
        x,_ = self.bilstm(x)
        x = self.attn(x)
        return self.fc(x)

# ==========================================================
# TRAIN FUNCTION
# ==========================================================

def train_hybrid(X_train_seq,y_train,X_test_seq,num_classes):

    model = HybridModel(X_train_seq.shape[2],num_classes).to(device)

    opt = optim.Adam(model.parameters(),lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train_seq,dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train,dtype=torch.long).to(device)

    for _ in range(80):
        opt.zero_grad()
        out = model(X_t)
        loss = loss_fn(out,y_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test_seq,
                                    dtype=torch.float32).to(device))
        preds = torch.argmax(logits,dim=1).cpu().numpy()

    return preds

# ==========================================================
# CROSS VALIDATION
# ==========================================================

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

top_k = 200

bal_scores=[]
acc_scores=[]

for fold,(tr,te) in enumerate(skf.split(X,y)):

    print("Fold:",fold+1)

    X_train,X_test=X[tr],X[te]
    y_train,y_test=y[tr],y[te]

    X_train,X_test=preprocess(X_train,X_test)

    # ---- ClassCDI ----
    scores=class_specific_cdi(X_train,y_train)
    idx=np.argsort(scores)[-top_k:]

    # ---- COH only ----
    idx=filter_coh(idx)

    X_train=X_train[:,idx]
    X_test=X_test[:,idx]
    feat_selected=feature_names[idx]

    # ---- Band structuring ----
    band_dict={b:[] for b in bands}
    for i,f in enumerate(feat_selected):
        for b in bands:
            if b in f.lower():
                band_dict[b].append(i)

    max_band=max([len(band_dict[b]) for b in bands])

    def build_seq(X_data):
        seq=[]
        for b in bands:
            idxs=band_dict[b]
            if len(idxs)==0:
                band=np.zeros((X_data.shape[0],max_band))
            else:
                band=X_data[:,idxs]
                if len(idxs)<max_band:
                    pad=max_band-len(idxs)
                    band=np.pad(band,((0,0),(0,pad)))
            seq.append(band)
        return np.stack(seq,axis=1)

    X_train_seq=build_seq(X_train)
    X_test_seq=build_seq(X_test)

    preds=train_hybrid(X_train_seq,
                       y_train,
                       X_test_seq,
                       num_classes=4)

    bal_scores.append(balanced_accuracy_score(y_test,preds))
    acc_scores.append(accuracy_score(y_test,preds))

print("\n===== FINAL RESULT =====")
print("Balanced Accuracy:",np.mean(bal_scores))
print("Accuracy:",np.mean(acc_scores))