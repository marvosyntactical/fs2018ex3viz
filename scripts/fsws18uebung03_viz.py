"""
help: python3 fsws18uebung03_viz.py -h

Formale und Computationelle Semantik WS2018
Uebung 3 Aufgabe 1


In dieser Aufgabe visualisierst du mithilfe von matplotlib oder einem anderen Visualisierungstool bzw. Modul deiner Wahl Attentionwerte
des Enhanced Sequential Inference Model für beispielhafte Entailment-Paare.

Für die vollständige Bearbeitung dieser Teilaufgabe ist lediglich die Funktion
				
				viz_entailment_pair()
				
zu implementieren. Sie wird von viz_input() aufgerufen, welche in der letzten Zeile von main() aufgerufen wird.
Für das Format deines Plots findest du im Aufgabenblatt ein unverbindliches Beispiel zur Orientierung.

Nach korrekter Implementation sollte dieses script über die Kommandozeile wie folgt aufgerufen werden können und einen MPL Graphen plotten:

$ python3 fsws18uebung03_viz.py --prem 'A cat is on a mat .' --hyp 'An animal is outside .'

-----------------------------------
Dependenzen:
python3.5+

torch
matplotlib.pyplot o.Ä.
wget

uvm...

es wird das Erstellen eines eigenen virtual environments mit sämtlichen benötigten Modulen im Projektordner empfohlen
-----------------------------------
"""
#Edited version of ./test_model.py
#dependencies not present in https://github.com/coetaur0/ESIM/ are as follows:
#esim/layer_attn_return.py
#esim/attn_return_model.py
#
#folder structure changed slightly as compared to above githup repo; esim code now in scripts/

#Author:
#	Marvin Koß
#	koss@cl.uni-heidelberg.de


import random
import os
import time
import pickle
import argparse
import torch
import json
from esim.attn_return_model import ESIM
from esim.utils import correct_predictions

#Your imports go here:



#Implement the following function
def viz_entailment_pair(premise, hypothesis, confusionmatrix):
	"""
	Visualizes attention matrix for entailment pair
	
	Args: 
		premise: premise in string form: "A cat is on a mat ."
		hypothesis: hypothesis in string form: "Tom is at the beach ."
		confusionmatrix: confusion matrix with attention values.
						 numpy matrix of size (hypothesis x premise)
	
	Returns:
		None
		(Matplotlib visualization with hypothesis on y axis and premise on x axis)
	"""
	raise NotImplementedError
	


def lookup(token):
	#looks up token in worddict, returning 1 (__OOV__ out of vocabulary) if not in vocab)
	if token in worddict.keys():
		return worddict[token]
	else:
		print("Warning: '{}' not in learned vocabulary.".format(token))
		return 1
		
def embed(sentence):
	#turn "A cat is on a mat ."esque sentence into tensor
	
	embedding = [2]#init with BOS
	embedding += [lookup(token) for token in sentence.split()]
	embedding += [3]#append EOS
	
	
	return torch.tensor(embedding)

def viz_input(model, premise, hypothesis):
	"""
	Visualize attention values for input premise and hypothesis in string form and return model prediction for input pair.
	Args:
		model: The torch module on which testing must be performed.
		dataloader: A DataLoader object to iterate over some dataset.
		premise: natural language premise string e.g. "A cat is on the mat ."
		hypothesis: natural language hypothesis string e.g. "Tom is at the beach ."
		
	Returns:
		probs: tensor of ESIM predictions for premise-hypothesis entailment (tensor sums to 1)
			tensor([entailed_prob, neutral_prob, contradiction_prob])
		
		(Matplotlib visualized attention values for premise-hypothesis)  
	"""	
	model.eval()
	device = model.device
	print("Model device is {}. Resuming on {}\n".format(device, device))
	
	#convert to tensors
	prem = embed(premise)
	hyp = embed(hypothesis)
		
	with torch.no_grad():
		
		prem_length = torch.tensor([prem.size(0)]).to(device)
		hyp_length = torch.tensor([hyp.size(0)]).to(device)
		
		prem_batch = prem.unsqueeze(0).to(device)
		hyp_batch = hyp.unsqueeze(0).to(device)
				
		_, probs, prem_hyp_attn, hyp_prem_attn = model(prem_batch, prem_length, hyp_batch, hyp_length)
				
		print("Model predictions for premise")
		print("\t" + premise)
		print("and hypothesis")
		print("\t" + hypothesis)
		print("\n{:10.3f} for entailment,\n{:10.3f} for neutral and \n{:10.3f} for contradiction.".format(float(probs[0][0]), float(probs[0][1]), float(probs[0][2])))
		
		viz_entailment_pair("BOS " + premise + " EOS", "BOS " + hypothesis + " EOS", hyp_prem_attn[0].numpy())
				
	return probs


def main(test_file,
         prem,
         hyp,
         pretrained_file,
         pp_config,
         vocab_size,
         embedding_dim,
         hidden_size=300,
         num_classes=3):
    """
    Test the ESIM model with pretrained weights on some dataset.
    Args:
        test_file: The path to a file containing preprocessed NLI data.
        prem: Premise string.
        hyp: Hypothesis string.
        pretrained_file: The path to a checkpoint produced by the
            'train_model' script.
        pp_config: Path to preprocessing config used in preprocessing for pretrained file.
        vocab_size: The number of words in the vocabulary of the model
            being tested.
        embedding_dim: The size of the embeddings in the model.
        hidden_size: The size of the hidden layers in the model. Must match
            the size used during training. Defaults to 300.
        num_classes: The number of classes in the output of the model. Must
            match the value used during training. Defaults to 3.
    """
    
    with open(os.path.normpath(pp_config), 'r') as cfg_file:
        ppcfg = json.load(cfg_file)
        
    targetdir = ppcfg["target_dir"]
    
    global worddict
    worddict = pickle.load(open(targetdir + "/worddict.pkl", "rb"))
    
    global inversedict
    inversedict = {v: k for k,v in worddict.items()}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")

    
    print("\t* Building model...")
    model = ESIM(vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_classes=num_classes,
                 device=device).to(device)

    #TODO make map location choosable
    checkpoint = torch.load(pretrained_file, map_location='cpu')
    
    model.load_state_dict(checkpoint['model'])

    #visualization function call, prem and hyp can be manually replaced here instead of input from console
    test_predictions = viz_input(model, prem, hyp)					
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the ESIM model on\
 some dataset')
    parser.add_argument('--checkpoint',
                        help="Path to a checkpoint with a pretrained model",
                        default="../data/checkpoints/best.pth.tar")
    parser.add_argument('--config', default='../config/test.json',
                        help='Path to a configuration file')
    parser.add_argument('--ppcfg', help='Path to config used in preprocessing for this model', default='../config/preprocessing.json')
    parser.add_argument('--prem', help="Premise in quotes a la 'A cat is on a mat .'", default='A cat is on a mat .')
    parser.add_argument('--hyp', help="Hypothesis in quotes a la 'An animal is outside .'", default='An animal is outside .')
    
    
    args = parser.parse_args()

    with open(os.path.normpath(args.config), 'r') as config_file:
        config = json.load(config_file)

    main(os.path.normpath(config['test_data']),
         args.prem,
         args.hyp,
         args.checkpoint,
         args.ppcfg,
         config['vocab_size'],
         config['embedding_dim'],
         config['hidden_size'],
         config['num_classes'])
