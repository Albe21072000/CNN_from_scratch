from moduli import *
import numpy as np

# Associo a ciasuna classe di immagini il relativo intero assegnato come label dai creatori del dataset
label_ind_by_names = {
    "Aeroplani": 0,
    "Automobili": 1,
    "Uccelli": 2,
    "Gatti": 3,
    "Cervi": 4,
    "Cani": 5,
    "Rane": 6,
    "Cavalli": 7,
    "Navi": 8,
    "Camion": 9,
}

# Metodo implementato dai fornitori del dataset Cifar-10 per importare i dati
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Importo i dati e ottengo i tensori per le immagini e per le relative classi
def importa_dati(directory):
    nomefil=r"\data_batch_"
    nomefil2=r"\test_batch"
    indirizzo_dati=directory+nomefil
    indirizzo_dati2=directory+nomefil2
    dati_ris=[]
    risp=[]
    for i in range(5):
        pr=unpickle(indirizzo_dati+str(i+1))
        X=pr.get(b'data')
        y=pr.get(b'labels')
        cont=0
        for i in X:
            sol=[i[0:1024],i[1024:2048],i[2048:3073]]
            dati_ris.append(np.array(sol).reshape(3,32,32))
            risp.append(y[cont])
            cont+=1
    pr=unpickle(indirizzo_dati2)
    X=pr.get(b'data')
    y=pr.get(b'labels')
    cont=0
    # Separo i tre canali RGB e li transormo in matrici quadrate usandu numpy
    for i in X:
        sol=[i[0:1024],i[1024:2048],i[2048:3073]]
        dati_ris.append(np.array(sol).reshape(3,32,32))
        risp.append(y[cont])
        cont+=1    
    return th.tensor(np.array(dati_ris)) , th.tensor((np.array(risp)))

# Metodo per separare l'insieme di training da quello di validation e test
def split_dataset(dati: th.Tensor, risposta: th.Tensor, prop_train):
    dim=dati.size()[0]
    # Genero casualmente una permutazione degli indici corrispondenti a ciascun immagine
    idx_rand = th.randperm(dim)
    dim_train=int(dim*prop_train) # Ricavo la dimensione dell'insieme di training
    idx_train=idx_rand[0:dim_train]  # Scelgo gli indici delle immagini da assegnare all'insieme di training
    idx_test=idx_rand[dim_train:dim] # I rimanenti andranno nell'altro dataset
    return dati[idx_train], risposta[idx_train],dati[idx_test], risposta[idx_test]

