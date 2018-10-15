import copy, time
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def printMetrics(dataLabels, dataConfidences, threshold):
    truePositives = np.sum((1-dataLabels) * (dataConfidences[:, 0]>threshold))
    predictedPositives = np.sum((dataConfidences[:, 0]>threshold))
    acutalPositives = np.sum((1-dataLabels))
    
    precision = (truePositives/predictedPositives) * 100
    recall = (truePositives/acutalPositives) * 100
    
    print(f'Confidence {threshold:.2f} | No_Change Precision: {precision:2.2f}% | No_Change Recall: {recall:2.2f}%')
    

def trainModel(model, criterion, optimizer, scheduler, dataLoaders, numEpochs):
    """
    Function to train a model
    """
    since = time.time()
    bestModelWts = copy.deepcopy(model.state_dict())
    bestAcc = 0.0
    for p in model.parameters(): p.requires_grad = True
    print(f'-->|{time.strftime("%X")}| Begin Training ...\n')

    for epoch in range(numEpochs):
        
        for phase in ['train', 'val']:
            runningLoss = 0.0; runningCorrects = 0
            if phase == 'train': scheduler.step(); model.train() 
            else: model.eval()

            for data in dataLoaders[phase]:
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.view(inputs.size(0)).cuda())
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train': loss.backward(); optimizer.step()

                runningLoss += loss.item()
                runningCorrects += torch.sum(preds == labels.data).item()

            epochLoss = runningLoss / (len(dataLoaders[phase]))
            epochAcc = runningCorrects / (len(dataLoaders[phase]) * inputs.size(0))
            print(f'-->|{time.strftime("%X")}| Epoch {epoch}/{numEpochs - 1} {phase} - Loss:{epochLoss:.4f} | Accuracy:{epochAcc*100:.2f}% | LR:{optimizer.param_groups[0]["lr"]:.1e}')

            # deep copy the model
            if phase == 'val' and epochAcc >= bestAcc: bestAcc = epochAcc; bestModelWts = copy.deepcopy(model.state_dict())
                
        print()

    timeElapsed = time.time() - since
    model.load_state_dict(bestModelWts)
    state = {'model':model, 'optimizer':optimizer}
    path = f'./modelWts/modelState_{time.strftime("%b%d-%H:%M")}_Acc_{bestAcc*100:.2f}%.pth'
    torch.save(state, path)
    print(f'Training complete in {(timeElapsed // 60):.0f}m {(timeElapsed % 60):.0f}s')
    print(f'Best val Acc: {bestAcc*100:.2f}%')
    return model



def testModel(model, dataLoader):
    """
    Function to test a model
    """
    since = time.time()
    print(f'-->|{time.strftime("%X")}| Begin Testing ...\n')

    runningCorrects = 0
    dataConfidences = np.array([]).reshape(0, 2)
    dataLabels = np.array([])
    for p in model.parameters(): p.requires_grad = False
    model.eval()

    for data in dataLoader:
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda(), requires_grad=False), Variable(labels.view(inputs.size(0)).cuda(), requires_grad=False)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        # print(labels); print(preds)
        runningCorrects += torch.sum(preds == labels.data).item()
        
        batchConfidences = F.softmax(outputs, dim=1).cpu().numpy()
        dataConfidences = np.append(dataConfidences, batchConfidences, 0)
        dataLabels = np.append(dataLabels, labels.cpu().numpy(), 0)

    epochAcc = runningCorrects / (len(dataLoader) * inputs.size(0))
    timeElapsed = time.time() - since
    print(f'Testing complete in {(timeElapsed // 60):.0f}m {(timeElapsed % 60):.0f}s')
    print(f'Acc: {epochAcc*100:.2f}%')
    printMetrics(dataLabels, dataConfidences, 0.5 )
    printMetrics(dataLabels, dataConfidences, 0.8)
    printMetrics(dataLabels, dataConfidences, 0.85)
    # printMetrics(dataLabels, dataConfidences, 0.90)
    print('Precision- % of correctly predicted no_change/total predicted no_change | Recall- % of correctly predicted no_change/total actual no_change')