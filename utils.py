import pennylane as qml
from pennylane import numpy as np
from config import config

qml.enable_tape()
num_words = config['NUM_WORDS']
qbits_per_word = config['QUBITS_PER_WORDS']

#wires = num_words*qbits_per_word

def Word_Shuffle_circuit(params, wires):
    """Apply a sequence of controlled-rotation
    params : 3 angles [0:2pi], 1 per qubit
    wires : the 3 wires indexing the current word"""
    for i in range(len(wires)):
        if i==len(wires)-1: #the last wire controls the first qubit
            qml.CRY(params[i], wires=[wires[i],wires[0]])
        else:
            qml.CRY(params[i], wires=[wires[i],wires[i+1]])
            
            
def Words_Loader_circuit(params, wires):
    """Load each vector with Word_Entangler_circuit
    all with the same set of parameters 'params'
    word_1 = wires[0:3], word_2 = wires[3:6], etc."""
    for i in range(len(wires)//qbits_per_word): #loop on each of the 3-qubits words
        i_ = i*qbits_per_word
        Word_Shuffle_circuit(params, wires[i_:i_+qbits_per_word])
        
        
def Words_Entangler_circuit(params_, wires):
    """Apply many controled-rotation between words
    'params_' is a matrix of different angles of size (num_words/2) x qbits_per_word """

    mask = np.zeros_like(params_)
    #EVEN number of qubits
    if num_words%2 == 0: 
        for bit in range(qbits_per_word):
            for i in range(int(np.ceil(num_words/2))): 
                if bit%2 != 0: 
                    i_ = i*qbits_per_word*2+qbits_per_word
                    if i_+bit+qbits_per_word > len(wires)-1: 
                        qml.CRY(params_[bit][i], wires=[wires[i_+bit],wires[bit]])
                        mask[bit][i]=1
                    else :
                        qml.CRY(params_[bit][i], wires=[wires[i_+bit],wires[i_+bit+qbits_per_word]])
                        mask[bit][i]=1
                else: 
                    i_ = i*qbits_per_word*2
                    qml.CRY(params_[bit][i], wires=[wires[i_+bit],wires[i_+bit+qbits_per_word]])
                    mask[bit][i]=1


    #ODD number of qubits            
    elif num_words%2 == 1: 
        for bit in range(qbits_per_word):
            if bit%2 == 0: #even bits: need to loop back the last CRot
                for i in range(int(np.ceil(num_words/2))):
                    i_ = i*qbits_per_word*2
                    if i_+bit+qbits_per_word > len(wires)-1: 
                        if bit == qbits_per_word-1 : #last layer
                            qml.CRY(params_[bit][i], wires=[wires[i_+bit],wires[bit]])
                            mask[bit][i]=1
                        else : 
                            qml.CRY(params_[bit][i], wires=[wires[i_+bit],wires[bit+1]])
                            mask[bit][i]=1
                    else :
                        qml.CRY(params_[bit][i], wires=[wires[i_+bit],wires[i_+bit+qbits_per_word]])
                        mask[bit][i]=1
            else: #odd bits
                for i in range(int(np.floor(num_words/2))):
                    i_ = i*qbits_per_word*2+qbits_per_word
                    qml.CRY(params_[bit][i], wires=[wires[i_+bit],wires[i_+bit+qbits_per_word]])
                    mask[bit][i]=1
                
def Ansatz_circuit(parameters, wires, num_layers):
    """apply sequentially the two layers
    Words_Loader_circuit followed by Words_Entangler_circuit
    'parameters' = [(param_loader,param_entangler) for layer in num_layers]"""
    for layer in range(num_layers):
        param_loader = parameters[layer][0]
        param_entangler = parameters[layer][1]
        Words_Loader_circuit(param_loader,wires)
        Words_Entangler_circuit(param_entangler,wires)
        
def circuit_final(parameters, wires, num_layers, target_word):
    """Here we include the extra word and the ancillary qubit 
    to compare the overlapping on one word of the sentence
    'target_word' : the index of the word to compare in [0:num_words-1]"""
    
    #wires segmentation
    wires_ansatz = wires[:-(qbits_per_word + 1)]
    wires_extra_word = wires[-(qbits_per_word + 1):-1]
    anc = wires[-1]
    wires_target_word = wires[int(target_word) * qbits_per_word : (int(target_word) + 1) * qbits_per_word]
    #print('wires_ansatz :',wires_ansatz)
    #print('wires_extra_word :',wires_extra_word)
    #print('anc :',anc)
    #print('wires_target_word :',wires_target_word)

    #apply variational circuit
    Ansatz_circuit(parameters, wires_ansatz, num_layers)
    
    #SWAP Test
    qml.Hadamard(wires=anc)
    for i in range(qbits_per_word):
        qml.CSWAP(wires=[anc, wires_extra_word[i], wires_target_word[i]])
    qml.Hadamard(wires=anc)
    
    
def encode_words(embeddings, indices):
    """The embeddings to amplitude encode into the circuit
    embeddings : embedding vectors stored by rows
    indices : position in the circuit, will be encoded in wires[qbits_per_word * index : qbits_per_word * (index+1)]"""
    for i,vec in enumerate(embeddings):
        qml.templates.embeddings.AmplitudeEmbedding(vec, range(indices[i]*qbits_per_word, (indices[i]+1)*qbits_per_word), pad=0.0, normalize=False)
