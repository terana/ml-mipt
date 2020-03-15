import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        self.title_conv = nn.Conv1d(hid_size, hid_size, padding=1, kernel_size=3)
        self.title_pool = nn.AdaptiveMaxPool1d(5)
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.full_conv = nn.Conv1d(hid_size, hid_size, kernel_size=7, padding=3)
        self.full_pool = nn.AdaptiveMaxPool1d(hid_size)
        
        self.category_out = nn.Linear(n_cat_features, hid_size)


        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2)
        self.final_dense = nn.Linear(in_features=hid_size*2, out_features=1)

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.title_pool(self.title_conv(title_beg))

        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full = self.full_pool(self.full_conv(full_beg))       

        category = F.relu(self.category_out(input3))  

        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        out = self.final_dense(F.relu(self.inter_dense(concatenated)))
        
        return out