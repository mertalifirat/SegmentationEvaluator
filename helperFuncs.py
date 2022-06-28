import mmcv
import torch
import numpy as np
import torchvision.transforms as T

def evaluate(model, dataset, dataloader):
    ### main evaluate function that will evaluate per image and will get result in overall
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    results = []
    dataset = dataloader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data,ann_path in dataloader:
        with torch.no_grad():
            data=data.to(device)
            result=model(data) 
            result=result.to('cpu') ### storing to cuda will full the gpu memory, so stored back to cpu
            ###  Resizing back to its original shape
            TR=T.Resize(dataset.originalShape)
            result=TR(result)
            ###  taking armax for detecting which pixel belongs to which class
            result=result[0].argmax(dim=0)
            ### Taking the annotated segmentation map
            seg_map=dataset.loadSegMap(ann_path[0])
            ### Evaluating the prediction and annotation
            resp=getValPerImg(result,seg_map,dataset.numClass)
            ### appending them to the results list
            results.append(resp)
            prog_bar.update()
    ### calculating the metrics
    precision,recall,IoU=metrics(results,dataset)
    return precision,recall,IoU

def getValPerImg(pred,label,numClass):
    #### comparing the annotation and prediction per image
    #### returns (area_intersect, area_union, area_pred_label, area_label)
    mask = (label != 255)
    pred = pred[mask]
    label = label[mask]
    label = torch.from_numpy(label)
    intersect = pred[pred == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(numClass), min=0, max=numClass - 1)
    area_pred_label = torch.histc(
        pred.float(), bins=(numClass), min=0, max=numClass - 1)
    area_label = torch.histc(
        label.float(), bins=(numClass), min=0, max=numClass - 1)
    area_union = area_pred_label + area_label - area_intersect
    return (area_intersect, area_union, area_pred_label, area_label)

def metrics(results,dataset):
    #### calculating the metrics with respect to results for all images
    results = tuple(zip(*results))
    total_area_intersect = sum(results[0])
    total_area_union = sum(results[1])
    total_area_pred_label = sum(results[2])
    total_area_label = sum(results[3])
    precision=total_area_intersect/total_area_pred_label
    recall=total_area_intersect/total_area_label
    IoU=total_area_intersect/total_area_union
    print("\n")
    for i in range(len(dataset.CLASSES)):
        print('\t Metric Values for {}'.format(dataset.CLASSES[i]))
        print('Precision \t Recall \t IoU ')
        print('{:.2f}        \t {:.2f}     \t {:.2f}'.format(100*precision[i],100*recall[i],100*IoU[i]))
    print('\t Metric Values for Avg')
    print('Precision \t Recall \t IoU ')
    print('{:.2f}        \t {:.2f}     \t {:.2f}'.format(100*precision.mean().item(),100*recall.mean().item(),100*IoU.mean().item()))
    return precision,recall,IoU
