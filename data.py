import torch
from torchvision import transforms
from torch.utils.data import Dataset
import sequencer_v2 as sq
import fcgrLib as fcgr
import pathlib
import numpy as np


class FcgrBags(Dataset):

    def __init__(self, kmers_len = 2, transform = None):
        #Estraiamo le sequenze dal file fasta
        path = str(pathlib.Path(__file__).resolve()).removesuffix("\data.py").replace('\\','/') + "/datasets/Drosophila_Promoter.fas"
        sequence_list = sq.sequencer(path)
        bags = []
        bags_lables = []
        for i in range(len(sequence_list)):
            #generiamo la rappresentazione in Frequency Chaos Game Representation(fcgr) e creiamo quattro bag per ogni immagine
            matrix = fcgr.getMatrice(sequence_list[i][0].lower(),k=kmers_len)
            m1 = matrix[0:2**(kmers_len-1),0:2**(kmers_len-1)]
            bags.append(m1)
            m2 = matrix[2**(kmers_len-1):,0:2**(kmers_len-1)]
            bags.append(m2)
            m3 = matrix[0:2**(kmers_len-1),2**(kmers_len-1):]
            bags.append(m3)
            m4 = matrix[2**(kmers_len-1):,2**(kmers_len-1):] 
            bags.append(m4)
            #Creiamo un etichetta per ognuna delle bag create
            if(sequence_list[i][1] == 'nuc'):
                bags_lables.extend(0 for _ in range(4))
            else:
                bags_lables.extend(1 for _ in range(4))

        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.fcgr_bags = torch.tensor(bags,device=device,dtype=torch.float32,requires_grad=True)
        self.lables = torch.tensor(bags_lables,device=device,dtype=torch.long)
        
        self.transform = transform
        self.sample_number = self.fcgr_bags.size()[0]

    def __getitem__(self,index):
        sample = self.fcgr_bags[index],self.lables[index]

        if self.transform != None:
            sample = (self.transform(sample[0],sample[1]))
        
        return sample
    
    def __len__(self):
        return self.sample_number