import torch as th
import torch.nn as nn 

# Se disponibile, sfrutto la GPU per eseguire il codice dove pssoibile
device = th.device("cuda" if th.cuda.is_available() else "cpu")
th.backends.cuda.matmul.allow_tf32 = True

# Questi moduli funzionano per immagini (o matrici) quadrate

def padd_img(immagini: th.tensor, dim_padd): # Uso il padding con valori casuali
    if dim_padd==0:  # Se non necessito il padding ritorno semplicemente la batch di immagini in input
        return immagini
    dim=immagini.size()[-1]
    # Genero la matrice campionando da una normale(0,1)
    fin = th.randn(immagini.size(0),immagini.size(1),dim+dim_padd*2, dim+dim_padd*2) 
    # Inserisco al centro della matrice generata quella originale
    fin[:,:,dim_padd:dim+dim_padd,dim_padd:dim+dim_padd]=immagini  
    return fin


class ReLU_p(nn.Module):   #Definisco la ReLU
    def __init__(self):
        super().__init__()
    def forward(self, dati: th.Tensor):
        return th.max(th.tensor(0.0), dati) # Ritorno per ciascun elemento delle matrice se stesso se positivo o 0 altrimenti


# Definisco un metodo che implementa il metodo forward della classe Pooling
def max_pool(mat_input, dim_kernel) -> th.Tensor:   # Implemento il pooling massimale
    dim_batch, num_canali, dim_img, _ = mat_input.size()
    # Calcolo le dimensioni dell'output
    dim_output = dim_img // dim_kernel
    # Creo un tensore di output inizializzato a zero
    output_tensor = th.zeros(dim_batch, num_canali, dim_output, dim_output)
    # Itero su ciascuna posizione dell'output
    for i in range(dim_output):
        for j in range(dim_output):
            # Estraggo ciascuna sottomatrice di cui voglio trovare il massimo dall'input
            input_slice = mat_input[:, :, i * dim_kernel:(i + 1) * dim_kernel, j * dim_kernel:(j + 1) * dim_kernel]
            # Trovo il massimo della sotomatrice e lo assegno alla relativa posizione della matrice di output
            output_tensor[:, :, i, j] = th.max(input_slice.reshape(dim_batch, num_canali,-1), dim=2)[0]  
    return output_tensor


class Pooling(nn.Module):

    def __init__(self, dim_kernel):  # Come parametro in input inserisco solo la dimensione della finestra di Pooling
        super().__init__()
        self.dim_kernel=dim_kernel
    
    def forward(self, dati: th.Tensor):
        # Applico il metodo visto sopra per il pooling massimale
        return max_pool(dati,self.dim_kernel).to(device)

# Implemento il Multi Layer Perceptron
class MLP(nn.Module):
    # Fornisco in input al costruttore il numero di dati in input e il numero di risultati in output
    def __init__(self,num_dati_input, num_out) :
        super().__init__()
        self.num_dati_input=num_dati_input
        self.num_out=num_out
        # Inizializzo i parametri del MLP campionando da una distribuzione uniforme
        self.W=nn.Parameter((th.rand(self.num_out,self.num_dati_input)-0.5)*2*(1/num_dati_input)**0.5)
        self.b=nn.Parameter((th.rand(self.num_out)-0.5)*2*(1/num_dati_input)**0.5)

    def forward(self, dati: th.Tensor):
        # Moltiplico i dati in input per i pesi e aggiungo il vettore dei bias
        return (dati.to(device) @ self.W.T)+self.b
        
# Metodo che simula l'unfold di pytorch 
def unfold(tensor, dim_kernel, stride):
    batch_size, channels, dim_in, _ = tensor.size()
    # Calcolo le dimensioni del tensore in uscita
    dim_uscita = (dim_in - dim_kernel) // stride + 1
    # Inizializzo il tensore unfolded in uscita a zero
    unfolded_tensor = th.zeros(batch_size, channels, dim_kernel, dim_kernel, dim_uscita, dim_uscita)
    # Riempio il tensore con i valori corrispondenti
    for i in range(dim_kernel):
        for j in range(dim_kernel):
            # Prendo gli elementi che verrano moltiplicati per lo stesso valore del kernel e li inserisco nello stesso vettore
            unfolded_tensor[:, :, i, j] = tensor[:, :, i:i + dim_uscita * stride:stride, j:j + dim_uscita * stride:stride]
    # Reshape per ottenere la forma bidimensionale
    unfolded_matrix = unfolded_tensor.reshape(batch_size, channels* dim_kernel**2, dim_uscita**2)
    return unfolded_matrix


# Classe che implementa il layer convolutivo
class Convoluzione(nn.Module):

    def __init__(self, canali_input, canali_out, dim_kernel=3, stride=1):
        super().__init__()
        self.dim_kernel = dim_kernel
        self.canali_input = canali_input
        self.canali_out = canali_out
        self.stride = stride
        # Inizializzo i parametri contenuti nei filtri convolutivi campionando da una distribuzione uniforme
        self.filtro = nn.Parameter((th.rand(canali_out, canali_input, dim_kernel, dim_kernel)-0.5)*(1/(canali_input*dim_kernel**2)))

    def forward(self, img):
        img = img.to(th.float32)
        dim_in= img.size()[-1]
        dim_batch = img.size(0)
        padding = 0
        # Verifico se, e nel caso quanto, padding è necessario 
        while (dim_in-self.dim_kernel+2*padding)/self.stride % 1 != 0:
            padding+=1
        img=padd_img(img,padding)
        # Calcolo le dimensioni delle matrici in output
        dim_out = (dim_in + 2 * padding - (self.dim_kernel - 1) - 1) // self.stride + 1
        # Eseguo l'unfold delle immagini in maniera da poterle moltiplicare,
        # dopo aver trasposto la seconda e terza dimensione,
        # per ciascun filtro contemporaneamente vettorializzando il codice 
        img_unf = unfold(img, self.dim_kernel,self.stride).to(device)
        out_unf = (img_unf.transpose(1, 2) @ self.filtro.view(self.filtro.size(0), -1).t()).transpose(1, 2)
        # Reshape per ottenere la forma corretta delle matrici di output
        return out_unf.reshape(dim_batch, self.canali_out, dim_out, dim_out)

# Implemento la Batch Normalization
class BatchNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # Parametri da apprendere
        self.w = th.nn.Parameter(th.ones(1,num_features,1,1))   
        self.b = th.nn.Parameter(th.zeros(1,num_features,1,1))
        # Creo i parametri in questo modo per non dover fare il reshape
        # per moltiplicarli con i dati in input

        # Media e varianza mobili
        # Inizializzati come parametri perché così sarà possibile estrarli
        # per ripristinare il modello originale quando attiverò l'early stopping
        # (questi parametri non dovranno essere considerati dalla backpropagation)
        self.media_mobile = th.nn.Parameter(th.zeros(1,num_features,1,1),requires_grad=False)
        self.varianza_mobile = th.nn.Parameter(th.ones(1,num_features,1,1),requires_grad=False)

    def forward(self, input):
        # Controllo se sono in fase di training, in caso positivo calcolo media e varianza per ogni canale della batch
        # e aggiorno media e varianza mobili, altrimenti uso tali valori per normalizzare i dati e non li aggiorno.
        if self.training:
            media_batch = input.mean(dim=(0, 2, 3), keepdim=True)
            var_batch = input.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            self.media_mobile = nn.Parameter((1 - self.momentum) * self.media_mobile + self.momentum * media_batch,requires_grad=False)
            self.varianza_mobile = nn.Parameter((1 - self.momentum) * self.varianza_mobile + self.momentum * var_batch,requires_grad=False)
        else:
            media_batch = self.media_mobile
            var_batch = self.varianza_mobile
        # Normalizzo l'input con media e varianza visti prima
        input_norm = (input - media_batch) / th.sqrt(var_batch + self.eps)
        # Moltiplico per i pesi e sommo il bias
        fin = self.w * input_norm + self.b
        return fin

# Implemento il dropout
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            # Generaro una matrice binaria per scegliere gli elementi da scartare
            # Campionando da un'uniforme (0,1) e controllando se il valore campionato è maggiore di p
            mask = th.rand(input.size()) > self.p
            # Pongo a zero i pixel scelti in precedenza
            output = input * mask.to(device) / (1 - self.p)
        else:
            # Durante la fase di test o validazione, restituisco semplicemente i dati in input
            output = input
        return output
    
# Modulo per "appiattire" un tensore composto da matrici multidimensionali in un vettore
class Flat(nn.Module):
    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x: th.Tensor):
        return x.reshape(x.size(0),-1)