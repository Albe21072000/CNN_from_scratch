
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
dati, y = importa_dati(r"Immagini") # Inserire il path assoluto se il file non viene trovato
y=y.to(th.long)
# Divido training e test set
dati_tr,y_tr,dati_test,y_test=split_dataset(dati,y,proporzione_training)

# Divido nuovamente il training dal validation test
dati_tr,y_tr,dati_val,y_val=split_dataset(dati_tr,y_tr,0.8)
#Standardizzo i dati usando il training set
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

# Alleno la quinta rete con gli iper-parametri selezionati
modello_allenato,tot_loss,tot_acc=trainCNN(rete_comp,1,tr_dataset,val_dataset,50,3,128)
acc_val,_=testCNN(modello_allenato,val_dataset)
print("Accuracy di validazione: "+str(acc_val))
acc_test,pred_test=testCNN(modello_allenato,test_dataset)
print("Accuracy di test: "+str(acc_test))

# Stampo l'heatmap dei risultati per il test set
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