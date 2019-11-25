import torch
# Download an example image from the pytorch website
import urllib
from PIL import Image as Image 
from torchvision import transforms
from IPython.display import Image as show_img
import numpy as np
import os
import string
import re
import pandas as pd
import pickle
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

def pre_process(lista, index):
    # remove directories
    lista = [item[index:] for item in lista]
    
    # remove digits
    lista = [re.sub(r'\b\d+\b', '', item) for item in lista]

    # remove hiffens 
    lista = [re.sub(r'\b-\b', ' ', item) for item in lista]

    # remove punctuation
    lista = [item.strip(string.punctuation) for item in lista]
    
    return lista

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf    

def convert_bars(lista):
    lista = [item.replace('\\', '/') for item in lista]
    return lista

def files_in_folder(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    files = convert_bars(files)
    return files

def predict(pill_transf, preprocess_transform, image):
    input_tensor = preprocess_transform(pill_transf(input_image))
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    scores = torch.nn.functional.softmax(output[0], dim=0)
    
    y_hat = scores.argmax() + 1 
    return y_hat.item(), scores

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    #batch =preprocess_transform(images).unsqueeze(0)
    logits = model(batch)
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def insertbest(list, n): 
    exist = 0
    # Searching for the position 
    for i in range(len(list)): 
        if list[i] < n: 
            index = i 
            exist = 1
            break
      
    # Inserting n in the list 
    if exist == 1:
        list = list[:i] + [n] + list[i:] 
        list.pop()
    else:
        index = None
    return list, index

def shiftbest(list, index, element):
    list = list[:index] + [element] + list[index:] 
    list.pop()
    return list

def insertworst(list, n): 
    exist = 0
    # Searching for the position 
    for i in range(len(list)): 
        if list[i] > n: 
            index = i 
            exist = 1
            break
      
    # Inserting n in the list 
    if exist == 1:
        list = list[:i] + [n] + list[i:] 
        list.pop()
    else:
        index = None
    return list, index

def shiftworst(list, index, element):
    list = list[:index] + [element] + list[index:] 
    list.pop()
    return list

# scores , id df , id image
# top 1 top 2 top 3 top 4 top 5
best5 = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]

# ultimo , penultimo, ...
worst5 = [[ 1.01, 1.01, 1.01, 1.01, 1.01], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]

directory = "C:/Users/Asus/Documents/IST/MECD 0101/AEP/Projeto/256_ObjectCategories"
dataset_input = [x[0] for x in os.walk(directory)]
dataset_input = pre_process(dataset_input, 71)

df = pd.read_csv('labels.csv',index_col = 'id')
dataset_csv = [df['label'][i] for i in range(len(df))]

filtered_input = [item for item in dataset_input if item in dataset_csv]

try: 
    model = pickle.load(open("model.pickle","rb"))
except:
    model = torch.hub.load('pytorch/vision:v0.4.2', 'mobilenet_v2', pretrained=True)
    model.eval()
    f = open('model.pickle', 'wb')
    pickle.dump(model, f)
    f.close()

directory = "C:/Users/Asus/Documents/IST/MECD 0101/AEP/Projeto/32 categories"
foldernames = [x[0] for x in os.walk(directory)]
foldernames.pop(0)
foldernames = convert_bars(foldernames)

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()
df = pd.read_csv('labels.csv',index_col = 'id')

try: 
    best5, worst5 = pickle.load(open("best and worst.pickle","rb"))

except:
    for folder, name in zip(foldernames, filtered_input):
            print("Directory: {}".format(folder))
            for i in range(len(best5)):
                print(best5[i])
            print("")
            for i in range(len(worst5)):
                print(worst5[i])

            files_names = files_in_folder(folder)
            image_index = 0
            for filename in files_names:
                with open(filename, "r") as file:
                    input_image = Image.open(filename)
                    image_index += 1
                    try: 
                        pred, scores = predict(pill_transf, preprocess_transform, input_image)
                        
                        index_truelab = df.loc[df["label"] == name].index[0] - 1
                        
                        # probability of model classifying the true label 
                        score_truelab = scores[index_truelab].item()

                        auxbest, idxbest = insertbest(best5[0], score_truelab)
                        auxworst, idxworst = insertworst(worst5[0], score_truelab)

                        if idxbest is not None:
                            best5[0] = auxbest
                            best5[1] = shiftbest(best5[1], idxbest, name)
                            best5[2] = shiftbest(best5[2], idxbest, df.loc[df.index == pred, 'label'].item())
                            best5[3] = shiftbest(best5[3], idxbest, image_index)
                        
                        if idxworst is not None:
                            worst5[0] = auxworst
                            worst5[1] = shiftworst(worst5[1], idxworst, name)
                            worst5[2] = shiftworst(worst5[2], idxworst, df.loc[df.index == pred, 'label'].item())
                            worst5[3] = shiftworst(worst5[3], idxworst, image_index)

                            
                    except RuntimeError: # black and white image
                        print("")
                        print("ERROR: black and white image")
                        continue


    for i in range(len(best5)):
        print(best5[i])
    print("")
    for i in range(len(worst5)):
        print(worst5[i])

    data = (best5, worst5)
    f = open('best and worst.pickle', 'wb')
    pickle.dump(data, f)
    f.close()

for i in range(len(best5)):
        print(best5[i])
print("")
for i in range(len(worst5)):
    print(worst5[i])