import pickle

def obtain_paths(folder_path):
    best5, worst5 = pickle.load(open("best and worst.pickle","rb"))

    directories= {"cannon": "029.cannon", "canoe":"030.canoe", "centipede":"034.centipede", "coffe mug":"041.coffee-mug",
                  "conch":"048.conch","electric guitar": "063.electric-guitar-101","football helmet":"076.football-helmet","golf ball":"088.golf-ball", "goose":"089.goose",
                  "harp":"098.harp", "hourglass":"110.hourglass", "hummingbird":"113.hummingbird", "llama":"134.llama-101",
                  "mushroom":"147.mushroom", "photocopier":"161.photocopier", "rifle":"173.rifle", "school bus":"178.school-bus",
                  "scorpion":"179.scorpion-101", "screwdriver":"180.screwdriver", "snail":"189.snail", "snowmobile":"192.snowmobile",
                  "soccer ball":"193.soccer-ball", "syringe":"210.syringe", "teapot":"212.teapot", "tennis ball": "216.tennis-ball",
                  "toaster":"220.toaster", "triceratops":"228.triceratops", "trilobite": "230.trilobite-101", "tripod":"231.tripod",
                  "umbrella":"235.umbrella-101", "wine bottle":"246.wine-bottle", "zebra":"250.zebra"}

    for i in range(100):
        best5[1][i] = directories[best5[1][i]] 
        if best5[3][i] < 10:
            best5[3][i] = best5[1][i][:3]+"_000"+str(best5[3][i])
        elif best5[3][i] >= 10 and best5[3][i] < 100:
            best5[3][i] = best5[1][i][:3]+"_00"+str(best5[3][i])
        else: 
            best5[3][i] = best5[1][i][:3]+"_0"+str(best5[3][i])

    for i in range(100):
        worst5[1][i] = directories[worst5[1][i]] 
        if worst5[3][i] < 10:
            worst5[3][i] = worst5[1][i][:3]+"_000"+str(worst5[3][i])
        elif worst5[3][i] >= 10 and worst5[3][i] < 100:
            worst5[3][i] = worst5[1][i][:3]+"_00"+str(worst5[3][i])
        else: 
            worst5[3][i] = worst5[1][i][:3]+"_0"+str(worst5[3][i])

    best_paths = []
    for i in range(100):
        filename = folder_path + "{}/{}.jpg".format(best5[1][i],best5[3][i])
        best_paths.append(filename)

    worst_paths = []
    for i in range(100):
        filename = folder_path + "{}/{}.jpg".format(worst5[1][i],worst5[3][i])
        worst_paths.append(filename)
    return best_paths, worst_paths
        
folder_path = "C:/Users/Asus/Documents/IST/MECD 0101/AEP/Projeto/32 categories/"
best100_paths, worst100_paths = obtain_paths(folder_path)