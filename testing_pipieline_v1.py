import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from typing import Dict, List

all_target_images = torch.randint(0, 2, (20,256, 256)).float()

all_target_masks = torch.randint(0, 2, (20,256, 256)).float()
all_target_instances = [[instance.cpu().numpy() for instance in extract_instances(target_mask)] for target_mask in all_target_masks]

iou_threshold = 0.8

def extract_instances(mask: np.ndarray) -> List[np.ndarray]:
    """
    Extract individual connected components (instances) from a binary mask.
    
    Parameters:
    mask (np.ndarray): Input binary mask of shape (H, W)
    
    Returns:
    List[np.ndarray]: List of binary masks, each containing one instance
    """
    labeled_mask = np.zeros_like(mask, dtype=int)
    label = 1
    instances = []
    height, width = mask.shape
    
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 1 and labeled_mask[y, x] == 0:
                component = []
                stack = [(y, x)]
                
                while stack:
                    cy, cx = stack.pop()
                    if labeled_mask[cy, cx] == 0:
                        labeled_mask[cy, cx] = label
                        component.append((cy, cx))
                        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] == 1 and labeled_mask[ny, nx] == 0:
                                stack.append((ny, nx))
                
                instance_mask = np.zeros_like(mask)
                for (iy, ix) in component:
                    instance_mask[iy, ix] = 1
                instances.append(instance_mask)
                label += 1
                
    return instances

def compute_iou(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    intersection = np.logical_and(target_mask, pred_mask).sum()
    union = np.logical_or(target_mask, pred_mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def compute_metrics(pred_masks: torch.Tensor, iter: int) -> Dict[str, float]:
    
    pred_instances = [instance for instance in extract_instances(pred_masks)]
    target_instances = all_target_instances[iter]
    
    all_ious = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_average_precisions = []
    
    for target_mask in target_instances:
        
        best_iou = 0
        argmax_best_iou = -1

        for pred_mask in pred_instances:
            iou = compute_iou(pred_mask, target_mask)
            if iou > best_iou:
                best_iou = iou
                argmax_best_iou = pred_mask
        
        tp = best_iou >= iou_threshold
        fp = best_iou < iou_threshold and best_iou > 0
        fn = target_mask.sum() > 0 and best_iou == 0
        
        all_ious.append(best_iou)
        
        if tp:
            precision, recall, f1, _ = 1.0, 1.0, 1.0, None
        else:
            precision, recall, f1, _ = 0.0, 0.0, 0.0, None
        
        target_mask_flat = target_mask.flatten()
        best_pred_mask_flat = argmax_best_iou.flatten() if best_iou > 0 else np.zeros_like(target_mask_flat)
        average_precision = average_precision_score(target_mask_flat, best_pred_mask_flat)
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)
        all_average_precisions.append(average_precision)
    
    metrics = {
        'Precision': np.mean(all_precisions),
        'Recall': np.mean(all_recalls),
        'Dice Score': np.mean(all_f1_scores),
        'Average Precision': np.mean(all_average_precisions),
        'Jaccard': np.mean(all_ious)
    }
    
    return metrics

def tester_func(pred_mask):
    
    predictions = model.predict(all_target_images,i)
    all_metrics = []

    for i in range(len(all_target_images)):
        all_metrics.append(compute_metrics(predictions[i]))
    
    
    average_metrics = {}

    for d in all_metrics:
        for key in d:
            if key in average_metrics:
                average_metrics[key] += d[key]
            else:
                average_metrics[key] = d[key]

    num_metrics = len(all_metrics)
    for key in average_metrics:
        average_metrics[key] /= num_metrics

    
    # Print results
    for metric, value in average_metrics.items():
        print(f"{metric}: {value:.4f}")
