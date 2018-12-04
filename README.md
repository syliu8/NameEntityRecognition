# NameEntityRecognition
The name entity recognition project is defined as "seeks to locate and classify named entity mentions in unstructured text into pre-defined categories such as the person names, organizations, locations, etc.". I have tested LSTM/BiLSTM with two embedding methods on the CoNLL-2003 shared task data files. Overfitting occurs as simple embedding and LSTM model does not accurately capture why some words combinations are the name entities. The further work to improve the NER classification would be considering capitalization feature, detecting words with LSTM and characters using character-level CNNs.

Reference: Jason P.C. Chiu, Eric Nichols. "Named Entity Recognition with Bidirectional LSTM-CNNs".arXiv:1511.08308

### Requirement
numpy

pandas

tqdm

tensorflow>=1.4.1

keras = 2.2.4

download glove.6B.100d.txt from https://nlp.stanford.edu/projects/glove/ and save to the dataset folder. 
put ner_train.txt and ner_validation to data foler
