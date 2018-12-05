import numpy as np
import pandas as pd
import tqdm
from keras.utils import Progbar

def readfile(filename):
    f = open(filename)
    sentences = []
    sentence = []

    # parse the sentences
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split(' ')
        sentence.append([splits[0],splits[-1]])

    if len(sentence) >0:
        sentences.append(sentence)
        sentence = []

    labelSet = set()
    words = {}

    # match the word and NER labels
    for sen in sentences:
        for token, label in sen:
            labelSet.add(label)
            words[token.lower()] = True

    label2Idx = {}
    for label in labelSet:
        label2Idx[label] = len(label2Idx)

    # embedding
    word2Idx = {}
    wordEmbeddings = []

    fEmbeddings = open("data/glove.6B.100d.txt", encoding="utf-8")

    for line in fEmbeddings:
        split = line.strip().split(" ")
        word = split[0]

        if len(word2Idx) == 0: 
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split)-1) 
            wordEmbeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split)-1)
            wordEmbeddings.append(vector)

        if split[0].lower() in words:
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings.append(vector)
            word2Idx[split[0]] = len(word2Idx)

    wordEmbeddings = np.array(wordEmbeddings)

    # create word2idx matrix    
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']    

    dataset = []

    wordCount = 0
    unknownWordCount = 0

    for sen in sentences:
        wordIndices = []    
        labelIndices = []
        for word,label in sen:  
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]                 
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1

            # get the label and map to int            
            wordIndices.append(wordIdx)
            labelIndices.append(label2Idx[label])

        dataset.append([wordIndices, labelIndices]) 

     # prepare data for training
    l = []
    for i in dataset:
        l.append(len(i[0]))
    l = set(l)
    data_proprocesed = []
    data_proprocesed_len = []
    z = 0
    for i in l:
        for j in dataset:
            if len(j[0]) == i:
                data_proprocesed.append(j)
                z += 1
        data_proprocesed_len.append(z)

    return data_proprocesed, data_proprocesed_len, wordEmbeddings, label2Idx
    
def iterate_minibatches(dataset,batch_len): 
    start = 0
    for i in batch_len:
        tokens = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t,l = dt
            l = np.expand_dims(l,-1)
            tokens.append(t)
            labels.append(l)
        yield np.asarray(labels),np.asarray(tokens)

# compute accuracy
def compute_f1(predictions, correct, idx2Label): 
    label_pred = []    
    for sentence in predictions:
        label_pred.append([idx2Label[element] for element in sentence])
        
    label_correct = []    
    for sentence in correct:
        label_correct.append([idx2Label[element] for element in sentence])
    
    prec = compute_precision(label_pred, label_correct)
    rec = compute_precision(label_correct, label_pred)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1

def compute_precision(guessed_sentences, correct_sentences):
    assert(len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0
    
    
    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        assert(len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            if guessed[idx][0] == 'B': 
                count += 1
                
                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctlyFound = True
                    
                    while idx < len(guessed) and guessed[idx][0] == 'I': 
                        if guessed[idx] != correct[idx]:
                            correctlyFound = False
                        
                        idx += 1
                    
                    if idx < len(guessed):
                        if correct[idx][0] == 'I': 
                            correctlyFound = False
                        
                    
                    if correctlyFound:
                        correctCount += 1
                else:
                    idx += 1
            else:  
                idx += 1
    
    precision = 0
    if count > 0:    
        precision = float(correctCount) / count
        
    return precision

def tag_dataset(dataset, model):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i,data in enumerate(dataset):    
        tokens, labels = data
        tokens = np.asarray([tokens])     
        pred = model.predict(tokens, verbose=False)[0]   
        pred = pred.argmax(axis=-1)            
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels