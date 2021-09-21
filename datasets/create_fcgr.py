import sequencer_v2 as sq
import fcgrLib as fcgr
import pathlib
import pickle

path = str(pathlib.Path(__file__).resolve()).removesuffix("\create_fcgr.py").replace('\\','/') + "/fasta/Drosophila_Promoter.fas"
print(path)

for k in range(2,5):
    sequence_list = sq.sequencer(path)
    for i in range(len(sequence_list)):
        sequence_list[i][0] = fcgr.getMatrice(sequence_list[i][0].lower(),k=2)
    path = str(pathlib.Path(__file__).resolve()).removesuffix("\create_fcgr.py").replace('\\','/') + "/data/fcgrk" + str(k) + ".pickle"
    with open(path,'wb') as f:
        pickle.dump(sequence_list,f)