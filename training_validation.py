from tqdm import tqdm
import numpy as np
import torch.utils.data as data
from moduli import *
import torch.optim as optim
from matplotlib import pyplot as plt


# Path dove salvo i pesi del modello con più accuratezza sul validation test per l'early stopping
loc_parametri = 'weights_temp.pt' 

# Introduco una classe che implementa l'Early Stopping
class EarlyStop:
    def __init__(self, pazienza=1, differenza_minima=0.001):  # Imposto i valori di pazienza e
        # di differenza di accuratezza minima per proseguire l'esecuzione
        self.pazienza = pazienza
        self.diff_min = differenza_minima
        self.cont = 0
        self.accuratezza_max = 0

    def valuta(self, accuratezza,modello: nn.Module):
        if accuratezza > self.accuratezza_max:
            self.accuratezza_max = accuratezza
            self.cont = 0
            th.save(modello.state_dict(),loc_parametri)  # Salvo lo stato del modello in un file esterno
        elif accuratezza < (self.accuratezza_max + self.diff_min):
            self.cont += 1
            if self.cont >= self.pazienza:
                return True
        return False

# Metodo per testare o validare la rete
def testCNN(modello: nn.Module, test_dataset):
    test_set_loader = data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    val_prev = []
    tot_correct = 0
    modello.eval()  # Indico al modello di essere in fase di test/validazione, in modo che non aggiorni i parametri
    with th.no_grad():
        for (batch_img, batch_pred) in tqdm(test_set_loader, desc='Test: '):
            classe_predetta = th.argmax(modello(batch_img), -1)
            val_prev.append(classe_predetta)
            tot_correct += th.sum(classe_predetta == batch_pred.to(device)).item()
    modello.train()
    return tot_correct/len(test_dataset), th.concatenate(val_prev)

# Metodo per allenare le reti
def trainCNN(modello, num_mod,tr_dataset, val_dataset, num_epoche, pazienza=3, dim_batch=128):
    modello.train()  # Indico al modello che siamo in fase di training
    opt=optim.Adam(modello.parameters(),0.001) # Uso AdAM come funzione di ottimizzazione
    tr_set_loader = data.DataLoader(tr_dataset, batch_size=dim_batch, shuffle=True)  # Divido il dataset
    # in batch di dimensione desiderata
    CE_loss = nn.CrossEntropyLoss() # Uso come loss la Cross Entropy
    early_stopping=EarlyStop(pazienza) # Imposto l'early stopping con la pazienza desiderata
    all_val_acc = []
    tr_loss=[]
    for e in range(num_epoche):
        loss_batch = []
        for (batch_img, batch_pred) in tqdm(tr_set_loader, desc='Training: '):
            opt.zero_grad() # Resetto i gradienti a 0
            classe_predetta = modello.forward(batch_img.to(device)) # Predico le immagini di training
            loss_val = CE_loss(classe_predetta, batch_pred.to(device)) # Calcolo la Cross Entropy Loss
            loss_val.backward()  # Calcolo i gradienti della loss rispetto ai parametri con la backpropagation
            opt.step()   # Aggiorno i parametri con i gradienti appena calcolati
            loss_batch.append(loss_val.detach().item())
        loss_tot = np.array(loss_batch).mean()
        val_accuracy, _ = testCNN(modello, val_dataset)  # Calcolo l'accuratezza di validazione
        all_val_acc.append(val_accuracy)
        tr_loss.append(loss_tot)
        print(f'Epoca: {e}\t|\tLoss Training: {loss_tot:0.5f}\t|\tAcc. validazione:{val_accuracy:0.2f}')
        stop=early_stopping.valuta(val_accuracy,modello) # Controllo se devo attivare l'early stopping
        if(stop):
            print("Early Stopping attivato!")
            break
    # Carico i parametri associati all'accuratezza di validazione migliore per la rete che abbiamo allenato
    modello.load_state_dict(th.load(loc_parametri)) 
    print("Accuratezza massima: " + str(max(all_val_acc)))
    # Grafici visibili se il codice è eseguito su notebook Jupyter
    fig,grafici=plt.subplots(2)
    grafici[0].plot(all_val_acc)
    grafici[0].set_title("Accuracy validazione modello "+str(num_mod))
    grafici[1].plot(tr_loss)
    grafici[1].set_title("Loss di training modello "+str(num_mod))
    return modello, tr_loss,max(all_val_acc)
    