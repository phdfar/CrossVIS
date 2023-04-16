import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
import json
from pycocotools import mask as masktools
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
import shutil
import glob
import pickle
import time
def decode_rle(seg):
    rle_mask = {
        "counts": seg['counts'].encode('utf-8'),
        "size": seg['size']
    }
    mask = np.ascontiguousarray(masktools.decode(rle_mask).astype(np.uint8))
    return mask

def calculate_iou(mask1, mask2):
    """
    Calculates the Intersection over Union (IoU) metric for two binary masks.

    :param mask1: A binary NumPy array representing a mask.
    :param mask2: A binary NumPy array representing another mask.
    :return: A float value between 0 and 1, representing the IoU between the two masks.
    """
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    if union == 0:
        return 0
    else:
        return intersection / union

def calculate_instance_segmentation_accuracy(gt_image, prediction_masks):
    """
    Calculates the instance segmentation accuracy for a given ground truth image and a list of prediction masks.

    :param gt_image: A numpy array representing the ground truth instance segmentation image, where each instance has a unique color.
    :param prediction_masks: A list of numpy arrays representing the binary masks for each prediction, along with their corresponding scores.
    :return: A tuple of two floats: the mean intersection over union (IoU) between the matched predictions and the ground truth instances, and the precision of the predictions.
    """
    # Convert the ground truth image to a list of binary masks for each instance
    gt_masks = []
    for i in np.unique(gt_image):
        if i!=0:
          mask = np.zeros_like(gt_image)
          mask[gt_image == i] = 1
          gt_masks.append(mask)
          
    # Calculate the IoU between each prediction and each ground truth instance
    iou_matrix = np.zeros((len(prediction_masks), len(gt_masks)))
    for i, (score, pred_mask) in enumerate(prediction_masks):
        for j, gt_mask in enumerate(gt_masks):
            #print('pred_mask',prediction_masks[i]['mask'].shape)
            iou_matrix[i, j] = calculate_iou(prediction_masks[i]['mask'], gt_mask)
            #print('###########')

    # Use the Hungarian algorithm to find the optimal assignment of predictions to ground truth instances
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)
    #print('row_indices, col_indices',row_indices, col_indices)

    # Calculate the mean IoU and precision for the matched predictions
    mean_iou = 0
    mean_iou_w = 0
    num_matches = 0
    num_true_positives = 0
    for i, j in zip(row_indices, col_indices):
        if iou_matrix[i, j] > 0:

            mean_iou += iou_matrix[i, j]
            mean_iou_w += prediction_masks[i]['score'] * iou_matrix[i, j]

            """
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(str(iou_matrix[i, j]))
            ax1.imshow(prediction_masks[i]['mask'])
            ax2.imshow(gt_masks[j])
            plt.figure()
            """

            num_matches += 1
            num_true_positives += 1
    if num_matches > 0:
        mean_iou /= num_matches
        mean_iou_w /=num_matches
        precision = num_true_positives / len(prediction_masks)
    else:
        mean_iou = 0
        mean_iou_w = 0
        precision = 0

    return mean_iou,mean_iou_w, precision

ls = glob.glob('/kaggle/working/CrossVIS/output/CrossVIS_R_50_1x/*.pth')

with open('/kaggle/working/CrossVIS/test_online.obj', 'rb') as fp:
  pthdict = pickle.load(fp)

names = list(pthdict.keys())
if names!=[]:
  for l in ls:
    sp = ls.split('/');sp=sp[-1]
    if sp not in names:
      torun = sp
else:
  torun = ls[0]

os.system('python /kaggle/working/CrossVIS/test_vis.py --config-file /kaggle/working/CrossVIS/configs/CrossVIS/R_50_1x.yaml --json-file /kaggle/working/CrossVIS/datasets/youtubevis/annotations/valid.json --opts MODEL.WEIGHTS /kaggle/working/CrossVIS/output/CrossVIS_R_50_1x/'+torun)
time.sleep(2)

outputname = '/kaggle/working/CrossVIS/results_'+torun+'.json'
shutil.move('/kaggle/working/CrossVIS/results.json',outputname)

print('saved ---> ',outputname )

with open(outputname,'r') as f:
  data =json.load(f)


with open('/kaggle/working/CrossVIS/online_test/valid_online.json','r') as f:
  valid =json.load(f)

gt_dict={}
allinstance={}

for v in range(len(valid['videos'])):
  gt_dict.update({valid['videos'][v]['id']:'/kaggle/working/CrossVIS/online_test/mask/mask/'+valid['videos'][v]['file_names'][0].replace('.jpg','.png')})
  allinstance.update({valid['videos'][v]['id']:[]})

for q in range(len(data)):
  id = data[q]['video_id']
  value = allinstance[id] + [q] 
  allinstance.update({id:value})

print('#######################################')
print('############### Comparison ############')

p=0;precision_T=0;mean_iou_T=0;mean_iou_w_T=0;
for v in range(len(valid['videos'])):

  prediction_masks=[]

  original_id = valid['videos'][v]['id']
  index = allinstance[original_id]
  gt_image = cv2.imread(gt_dict[original_id],0)

  for q in index:
    img = data[q]['segmentations'][0]
    if img!=None:
      img = decode_rle(img)
    else:
      img = np.zeros((720,1280))

    prediction_masks.append({'score':data[q]['score'],'mask':img})

  mean_iou,mean_iou_w, precision = calculate_instance_segmentation_accuracy(gt_image, prediction_masks)

  mean_iou_T = mean_iou_T + mean_iou
  mean_iou_w_T = mean_iou_w_T + mean_iou_w
  precision_T = precision_T + precision
  p=p+1;
  #if p==3:
    #break
a = mean_iou_T/p
b = mean_iou_w_T/p
c = precision_T/p

print('mean_iou',a)
print('mean_iou_w',b)
print('precision',c)

pthdict.update({torun:[a,b,c]})

with open('/kaggle/working/CrossVIS/test_online.obj', 'wb') as fp:
  pickle.dump(pthdict, fp)

print('#######################################')
print('############### END Comparison ############')
time.sleep(2)
