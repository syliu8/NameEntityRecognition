# NameEntityRecognition
The name entity recognition project is defined as "seeks to locate and classify named entity mentions in unstructured text into pre-defined categories such as the person names, organizations, locations, etc.". I have tested LSTM/BiLSTM with two embedding methods on the CoNLL-2003 shared task data files. Overfitting occurs as simple embedding and LSTM model does not accurately capture why some words combinations are the name entities. The further work to improve the NER classification would be considering capitalization feature, detecting words with LSTM and characters using character-level CNNs.

Reference: Jason P.C. Chiu, Eric Nichols. "Named Entity Recognition with Bidirectional LSTM-CNNs".arXiv:1511.08308
