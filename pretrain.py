

import logging

from transformers import AutoTokenizer
import json
import tqdm

from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
import logging

from sklearn.preprocessing import LabelBinarizer
from pytorch_lightning import Trainer
import pandas as pd
import numpy as np
import re

# Huggingface transformers
import transformers
from transformers import BertModel,BertTokenizer,AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn ,cuda
from torch.utils.data import DataLoader,Dataset,RandomSampler, SequentialSampler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#handling html data
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.ERROR)




base_path = '/home/chaofeng/Documents/vscode/finsim/data/kg'
trainfile = base_path + "/train.json"

raw_datasets = pd.read_json(trainfile)


mlb = LabelBinarizer()
y = raw_datasets.label.tolist()
yt = mlb.fit_transform(y)
x = raw_datasets.term.tolist()
x_train,x_test,y_train,y_test = train_test_split(x, yt , test_size=0.1, random_state=42,shuffle=True)

x_tr,x_val,y_tr,y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42,shuffle=True)



class FinsimDataset (Dataset):
    def __init__(self,quest,tags, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = quest
        self.labels = tags
        self.max_len = max_len
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item_idx):
        text = self.text[item_idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True, # Add [CLS] [SEP]
            max_length= self.max_len,
            padding = 'max_length',
            return_token_type_ids= False,
            return_attention_mask= True, # Differentiates padded vs normal token
            truncation=True, # Truncate data beyond max length
            return_tensors = 'pt' # PyTorch Tensor format
          )
        
        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()
        #token_type_ids = inputs["token_type_ids"]
        
        return {
            'input_ids': input_ids ,
            'attention_mask': attn_mask,
            'label': torch.tensor(self.labels[item_idx], dtype=torch.float)
            
        }


class FinsimDataModule (pl.LightningDataModule):    
    def __init__(self,x_tr,y_tr,x_val,y_val,x_test,y_test,tokenizer,batch_size=16,max_token_len=200):
        super().__init__()
        self.tr_text = x_tr
        self.tr_label = y_tr
        self.val_text = x_val
        self.val_label = y_val
        self.test_text = x_test
        self.test_label = y_test
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

    def setup(self):
        self.train_dataset = FinsimDataset(quest=self.tr_text,  tags=self.tr_label,tokenizer=self.tokenizer,max_len= self.max_token_len)
        self.val_dataset= FinsimDataset(quest=self.val_text, tags=self.val_label,tokenizer=self.tokenizer,max_len = self.max_token_len)
        self.test_dataset =FinsimDataset(quest=self.test_text, tags=self.test_label,tokenizer=self.tokenizer,max_len = self.max_token_len)
        
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size= self.batch_size, shuffle = True , num_workers=4)

    def val_dataloader(self):
        return DataLoader (self.val_dataset,batch_size= 16)

    def test_dataloader(self):
        return DataLoader (self.test_dataset,batch_size= 16)


tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
BERT_MODEL_NAME = "ProsusAI/finbert"
N_EPOCHS = 12
BATCH_SIZE = 32
MAX_LEN = 100
LR = 2e-05
Bert_tokenizer = tokenizer
datamodule = FinsimDataModule(x_tr,y_tr,x_val,y_val,x_test,y_test,Bert_tokenizer,BATCH_SIZE,MAX_LEN)
datamodule.setup()




class FinsimClassifier(pl.LightningModule):
    # Set up the classifier
    def __init__(self,n_classes=17,steps_per_epoch=None,n_epochs=3, lr=2e-5):
        super().__init__()

        self.bert=BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier =self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
            nn.ReLU()
        )
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self,input_ids, attn_mask):
        output = self.bert(input_ids=input_ids,attention_mask=attn_mask)
        output = self.classifier(output.pooler_output)
                
        return output

    def training_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        outputs = self(input_ids,attention_mask)
        loss = self.criterion(outputs,labels)
        self.log('train_loss',loss , prog_bar=True,logger=True)
        
        return {"loss" :loss, "predictions":outputs, "labels": labels }


    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        outputs = self(input_ids,attention_mask)
        loss = self.criterion(outputs,labels)
        self.log('val_loss',loss , prog_bar=True,logger=True)        
        return loss

    def test_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        outputs = self(input_ids,attention_mask)
        loss = self.criterion(outputs,labels)
        self.log('test_loss',loss , prog_bar=True,logger=True)
        
        return loss
    
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters() , lr=self.lr)
        warmup_steps = self.steps_per_epoch//3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,total_steps)

        return [optimizer], [scheduler]


steps_per_epoch = len(x_tr)//BATCH_SIZE
model = FinsimClassifier(n_classes=17, steps_per_epoch=steps_per_epoch,n_epochs=N_EPOCHS,lr=LR)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',# monitored quantity
    filename='QTag-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3, #  save the top 3 models
    mode='min', # mode of the monitored quantity  for optimization
)

trainer = Trainer(max_epochs = N_EPOCHS , gpus = 1, callbacks=[checkpoint_callback],progress_bar_refresh_rate = 30)
trainer.fit(model, datamodule)



model_path = checkpoint_callback.best_model_path
print(model_path)

trainer.test(model,datamodule=datamodule)

# finberttokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# finbert =  AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=17)


# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)




# labels ={0: "Bonds",1: "Forward", 2: "Funds", 3: "Future", 4: "MMIs", 
#         5: "Option", 6: "Stocks",7: "Swap",8: "Equity Index",9: "Credit Index",
#         10: "Securities restrictions",11: "Parametric schedules",12: "Debt pricing and yields",
#         13: "Credit Events",14: "Stock Corporation",15: "Central Securities Depository",16: "Regulatory Agency"}

# max_seq_length=512
# device='cuda:1'