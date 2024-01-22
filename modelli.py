from moduli import *

# Definisco la prima "semplice" rete 
l1_base=nn.Sequential(Convoluzione(3,8,3),ReLU_p(),Pooling(2))
l2_base=nn.Sequential(Convoluzione(8,16,3),ReLU_p(),Pooling(2))
l3_base=nn.Sequential(Convoluzione(16,16,3),ReLU_p(),Pooling(2))
fin_l_base=nn.Sequential(Flat(), MLP(64,10))
rete_base=nn.Sequential(l1_base,l2_base,l3_base,fin_l_base).to(device=device)

# Aumento il numero di layer convolutivi e di canali in output
l1=nn.Sequential(Convoluzione(3,32,3),ReLU_p(),Convoluzione(32,64,3),Pooling(2))
l2=nn.Sequential(Convoluzione(64,128,3),ReLU_p(),Convoluzione(128,128,3),Pooling(2))
l3=nn.Sequential(Convoluzione(128,256,3),ReLU_p())
fin_l=nn.Sequential(Flat(), MLP(2304,1024),ReLU_p(),MLP(1024, 512),ReLU_p(),MLP(512, 10))
rete_256=nn.Sequential(l1,l2,l3,fin_l).to(device=device)

# Aggiugno la batch normalization
l1=nn.Sequential(Convoluzione(3,32,3),BatchNormalization(32),ReLU_p(),Convoluzione(32,64,3),Pooling(2))
l2=nn.Sequential(Convoluzione(64,128,3),BatchNormalization(128),ReLU_p(),Convoluzione(128,128,3),Pooling(2))
l3=nn.Sequential(Convoluzione(128,256,3),BatchNormalization(256),ReLU_p())
fin_l=nn.Sequential(Flat(), MLP(2304,1024),ReLU_p(),MLP(1024, 512),ReLU_p(),MLP(512, 10))
rete_batch_norm=nn.Sequential(l1,l2,l3,fin_l).to(device=device)

# Aggiungo il dropout ma non la batch normalization
l1=nn.Sequential(Convoluzione(3,32,3),ReLU_p(),Convoluzione(32,64,3),Pooling(2))
l2=nn.Sequential(Convoluzione(64,128,3),ReLU_p(),Convoluzione(128,128,3),Pooling(2),Dropout(0.05))
l3=nn.Sequential(Convoluzione(128,256,3),ReLU_p(),Dropout(0.1))
fin_l=nn.Sequential(Flat(), MLP(2304,1024),ReLU_p(),MLP(1024, 512),ReLU_p(),Dropout(0.1),MLP(512, 10))
rete_dropout=nn.Sequential(l1,l2,l3,fin_l).to(device=device)

# Implemento la rete con sia batch normalization che dropout
l1=nn.Sequential(Convoluzione(3,32,3),BatchNormalization(32),ReLU_p(),Convoluzione(32,64,3),Pooling(2))
l2=nn.Sequential(Convoluzione(64,128,3),BatchNormalization(128),ReLU_p(),Convoluzione(128,128,3),Pooling(2),Dropout(0.05))
l3=nn.Sequential(Convoluzione(128,256,3),BatchNormalization(256),ReLU_p(),Dropout(0.1))
fin_l=nn.Sequential(Flat(), MLP(2304,1024),ReLU_p(),MLP(1024, 512),ReLU_p(),Dropout(0.1),MLP(512, 10))
rete_comp=nn.Sequential(l1,l2,l3,fin_l).to(device=device)

# Implemento la stessa rete di prima che verrà validata con dimensione di batch pari a 256
l1=nn.Sequential(Convoluzione(3,32,3),BatchNormalization(32),ReLU_p(),Convoluzione(32,64,3),Pooling(2))
l2=nn.Sequential(Convoluzione(64,128,3),BatchNormalization(128),ReLU_p(),Convoluzione(128,128,3),Pooling(2),Dropout(0.05))
l3=nn.Sequential(Convoluzione(128,256,3),BatchNormalization(256),ReLU_p(),Dropout(0.1))
fin_l=nn.Sequential(Flat(), MLP(2304,1024),ReLU_p(),MLP(1024, 512),ReLU_p(),Dropout(0.1),MLP(512, 10))
rete_comp_batch256=nn.Sequential(l1,l2,l3,fin_l).to(device=device)

# Aggiungo anche un ultimo modello in cui inserisco il dropout sui pixel delle matrici in input
l1=nn.Sequential(Dropout(0.02),Convoluzione(3,32,3),BatchNormalization(32),ReLU_p(),Convoluzione(32,64,3),Pooling(2))
l2=nn.Sequential(Convoluzione(64,128,3),BatchNormalization(128),ReLU_p(),Convoluzione(128,128,3),Pooling(2),Dropout(0.025))
l3=nn.Sequential(Convoluzione(128,256,3),BatchNormalization(256),ReLU_p(),Dropout(0.1))
fin_l=nn.Sequential(Flat(), MLP(2304,1024),ReLU_p(),MLP(1024, 512),ReLU_p(),Dropout(0.1),MLP(512, 10))
rete_comp_drop=nn.Sequential(l1,l2,l3,fin_l).to(device=device)

# Inserisco tutti i modelli da validare in una lista
modelli=[rete_base,rete_256,rete_batch_norm,rete_dropout,rete_comp,rete_comp_batch256,rete_comp_drop]