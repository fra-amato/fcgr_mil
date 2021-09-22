import sys
def sequencer(fname):
    dna_type = []   #lista che contiene l'informazione riguardante 'nuc' o 'link'.
    sequence = []   #lista contenente le diverse sequenze di dna.
    dna_list = []
    temp = ''
    with open(fname,'r')as f:

        for line in f:  #lettura del file riga per riga.
            line=line.strip()
            if line.startswith('>'):
                header=line.split() #tramite il metodo .split spezzo le righe d'intestazione in corrispondenza degli spazi vuoti.
                dna_type.append(header[-1])

            else:
                temp=temp+line  #uso una stringa temporanea come storage per la sequenza di dna, che è la stessa anche quando si va a capo.
                if not line.strip(): #quando il ciclo incontra una riga vuota, la sequenza di dna viene "appesa" alla lista e temp resettata.

                    sequence.append(temp)
                    temp=''
                    continue
        dna_list=[list(a) for a in zip(sequence,dna_type)] #mappatura dell'iteratore in lista
    return(dna_list) #combino le due liste per ottenere la lista [seq,nuc/link]

if __name__ == "__main__":
    argument = sys.argv[1]
    sequencer(argument)