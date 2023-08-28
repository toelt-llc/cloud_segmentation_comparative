from sklearn.metrics import confusion_matrix
# Libraries for model evaluation and validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score, f1_score
from keras import backend as K

from sklearn.metrics import accuracy_score, jaccard_score
import numpy as np

# ###precision---
# def precision(gt, mask):
#     gt = gt.flatten()
#     mask = mask.flatten()
#     tn, fp, fn, tp = confusion_matrix(gt, mask, labels=[0, 1]).ravel()
#     prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     return prec

# ####recall---
# def recall(gt,mask):
#     gt = gt.flatten()
#     mask = mask.flatten()
#     tn, fp, fn, tp = confusion_matrix(gt, mask, labels=[0, 1]).ravel()
#     rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     return rec

# ###f1 score--
# def f1_score(prec,rec):
#     if (prec + rec) > 0:
#         f1 = 2 * (prec * rec) / (prec + rec)
#     else:
#         f1 = 0.0
#     return f1

# ### jaccard 
# def jaccard(gt,mask):
#     gt = gt.flatten()
#     mask = mask.flatten()
#     tn, fp, fn, tp = confusion_matrix(gt, mask, labels=[0, 1]).ravel()
#     jaccard_index = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0
#     return jaccard_index


def Overall(gt,mask):
    gt = gt.flatten()
    mask = mask.flatten()
    tn, fp, fn, tp = confusion_matrix(gt, mask, labels=[0, 1]).ravel()
    overall_accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return overall_accuracy

###aji score
def get_fast_aji(true, pred):
    
    true = np.copy(true) # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None,]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [None,]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
    
    # prefill with value
    pairwise_inter = np.zeros([len(true_id_list) -1, 
                               len(pred_id_list) -1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) -1, 
                               len(pred_id_list) -1], dtype=np.float64)

    # caching pairwise
    for true_id in true_id_list[1:]: # 0-th is background
        t_mask = true_masks[int(true_id)]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0: # ignore
                continue # overlaping background
            p_mask = pred_masks[int(pred_id)]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[int(true_id)-1, int(pred_id)-1] = inter
            pairwise_union[int(true_id)-1, int(pred_id)-1] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care 
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()
    #
    paired_true = (list(paired_true + 1)) # index to instance ID
    paired_pred = (list(paired_pred + 1))
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score


# def print_m(Y_test, preds_test_t):
#     sum_m = 0
#     for i in range(len(Y_test)):
#         sum_m = sum_m + precision(Y_test[i],preds_test_t[i])
#     prec = sum_m/len(Y_test)

#     sum_m = 0
#     for i in range(len(Y_test)):
#         sum_m = sum_m + recall(Y_test[i],preds_test_t[i])
#     rec = sum_m/len(Y_test)

#     sum_m = 0
#     for i in range(len(Y_test)):
#         sum_m = sum_m + jaccard(Y_test[i],preds_test_t[i])
#     jaccard1 = sum_m/len(Y_test)


#     sum_m = 0
#     for i in range(len(Y_test)):
#         sum_m = sum_m + Overall(Y_test[i],preds_test_t[i])
#     Overall1 = sum_m/len(Y_test)


#     f1 = f1_score(prec,rec)

#     aji = get_fast_aji(Y_test,preds_test_t)

#     print("Jaccard Index", jaccard1)
#     print("final f1", f1)
#     print("final precision",prec)
#     print("final recall",rec)
#     print("Overall Accuracy", Overall1)
#     print("Average Jaccard Index", aji)


def avg_metrics(Y_test, preds_test_t):
    sum_m = 0
    for i in range(len(Y_test)):
        sum_m = sum_m + Overall(Y_test[i],preds_test_t[i])
    Overall1 = sum_m/len(Y_test)

    aji = get_fast_aji(Y_test,preds_test_t)

    return Overall1, aji

# ------------------------------------------------------

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def IoU(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return intersection / union


def print_metrics(y_true, y_pred):
    # Cast to float types
    y_true_f = y_true.astype('float32')
    y_pred_f = y_pred.astype('float32')
    
    y_true_flatten = y_true_f.reshape(-1)
    y_pred_flatten = y_pred_f.reshape(-1)
    
    print("Pixel Accuracy: ", round(accuracy_score(y_true_flatten, y_pred_flatten), 4))
    print("Precision: ", round(precision_score(y_true_flatten, y_pred_flatten), 4))
    print("Recall: ", round(recall_score(y_true_flatten, y_pred_flatten), 4))
    print("F1 Score: ", round(f1_score(y_true_flatten, y_pred_flatten), 4))
    print("AUC Score: ", round(roc_auc_score(y_true_flatten, y_pred_flatten), 4))

    print("-------------------------------------")
    
    print("Dice Coefficient: ", round(K.eval(dice_coef(y_true_f, y_pred_f)), 4))
    
    # Jaccard Score
    
#     print("IoU (Jaccard Coefficient): ", round(K.eval(IoU(y_true_f, y_pred_f)), 4))
    
#     print("Jaccard Coefficient: ", round(jaccard(y_true_f, y_pred_f), 4))

    print("Jaccard Score (IoU): ", round(jaccard_score(y_true_flatten, y_pred_flatten), 4))

    print("-------------------------------------")

    aji, o_acc = avg_metrics(y_true_f, y_pred_f)

    print("Overall Accuracy", round(o_acc, 4))
    print("Average Jaccard Index", round(aji, 4))
    

def find_best_threshold(prob_mask, y_true):
    best_threshold = 0
    best_iou = 0

    # Ensuring y_true is in correct data type
    y_true = y_true.astype('float32')
    
    for threshold in np.arange(0.0, 1.1, 0.1):  # You can adjust the range and step size as needed
        binary_mask = (prob_mask > threshold).astype('float32')
        
        assert binary_mask.shape == y_true.shape, "Shape mismatch between binary_mask and y_true!"
        
        # Compute Jaccard Score (IoU)
        iou_score = jaccard_score(y_true.flatten(), binary_mask.flatten())

        if iou_score > best_iou:
            best_iou = iou_score
            best_threshold = threshold

        print(f'Threshold: {threshold:.1f}, IoU: {iou_score:.4f}')

    print(f'\nBest Threshold: {best_threshold:.1f}, Best IoU: {best_iou:.4f}')
    return best_threshold