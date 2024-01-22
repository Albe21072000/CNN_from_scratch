# Progetto per l'esame di Computational Learning di Alberto Biliotti
# Matricola: 7109894

from moduli import *
from gestione_dati import *
from training_validation import *
import random as rd
from modelli import *
import seaborn as sns
import matplotlib as mpl

proporzione_training=0.7 # specifico la proporzione di dati che entrerà nel Training Set sul totale
# Imposto il seed per la riproducibilità
rd.seed(21072000)
th.manual_seed(21072000)
# Importo i dati
dati, y = importa_dati(r"Immagini")  # Inserire il path assoluto se il file non viene trovato
y=y.to(th.long)
# Divido training e test set
dati_tr,y_tr,dati_test,y_test=split_dataset(dati,y,proporzione_training)
# Divido nuovamente il training dal validation test
dati_tr,y_tr,dati_val,y_val=split_dataset(dati_tr,y_tr,0.8)

#Standardizzo tutti i dati usando la media e la deviazione standard calcolati solo sul training set
dati_tr=dati_tr.float()
medie=dati_tr.mean(dim=(0,2,3),keepdim=True)
dev_stand=dati_tr.std(dim=(0,2,3),keepdim=True)
dati_tr_stand=(dati_tr-medie)/dev_stand
dati_val_stand=(dati_val-medie)/dev_stand
dati_test_stand=(dati_test-medie)/dev_stand

# Trasferisco i miei dati in un dataset per gestire meglio le batch 
tr_dataset = data.TensorDataset(dati_tr_stand, y_tr)
val_dataset = data.TensorDataset(dati_val_stand, y_val)
test_dataset=data.TensorDataset(dati_test_stand, y_test)

# Inizio la fase di model selection
cont=1  
vettore_acc=[]  # Questa listà conterrà le accuracy di validazione di tutti i modelli
mod_allenati=[] # Qui invece mantengo tutti i modelli già addestrati
dim_batch=[128,128,128,128,128,256,256]   # Qui specifico le dimensioni della batch con cui verrà addestrato ciascun modello
for rete in modelli:
    print("Alleno la rete numero: ",cont)
    modello_allenato,tot_loss,acc_max=trainCNN(rete,cont,tr_dataset,val_dataset,50,3,dim_batch[cont-1])
    cont+=1
    vettore_acc.append(acc_max)
    mod_allenati.append(modello_allenato)
# Ora scelgo il modello con la maggior accuratezza sull'insieme di validazione
ind_mod_migliore=vettore_acc.index(max(vettore_acc))
print("Scelgo il modello numero "+ str(ind_mod_migliore+1))
mod_scelto=mod_allenati[ind_mod_migliore]
print(vettore_acc)
# Qui eseguo il model assessment, valutando le prestazioni della rete scelta sull'insieme di test
acc_test,pred_test=testCNN(mod_scelto,test_dataset)
print("Accuracy di test: "+str(acc_test))
# Mostro l'heatmap per il test set
confusion_mat = th.sparse_coo_tensor(indices=th.stack((pred_test.cpu(),y_test.cpu())), values=th.ones_like(pred_test.cpu()), size=(10,10), dtype=th.int).to_dense()
plt.plot()
mpl.rcParams['figure.dpi'] = 100
ax=plt.subplot()
plt.rcParams.update({'font.size': 5})
sns.heatmap(confusion_mat/th.sum(confusion_mat,0,keepdim=True))
plt.xlabel('Classi vere')
plt.ylabel('Classi predette')
ax.xaxis.set_ticklabels(label_ind_by_names) 
ax.yaxis.set_ticklabels(label_ind_by_names);
# Per visualizzare tutti i grafici correttamente conviene lanciare l'esecuzione di questo file da un notebook Jupyter con il comando:
# from main import *