import torch
import torch.nn as nn
from tqdm import tqdm

from collections import OrderedDict
from functools import partial
from backbone.backbone import build_resnet_fpn_backbone
from config import cfg
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


    
def loadbackbone(device):
    backbone_type = cfg.MODEL.BACKBONE.TYPE
    print('backbone_type',backbone_type)
    backbone = build_resnet_fpn_backbone(cfg)

    info_to_print = [
        "Backbone type: {}".format(cfg.MODEL.BACKBONE.TYPE),
        "Backbone frozen: {}".format("Yes" if cfg.TRAINING.FREEZE_BACKBONE else "No")
    ]

    pretrained_wts_file = os.getcwd()+'/mask_rcnn_R_101_FPN_backbone.pth'
    #print_fn("Restoring backbone weights from '{}'".format(pretrained_wts_file))

    if os.path.exists(pretrained_wts_file):
        print('Backbone Loaded From ',pretrained_wts_file)
        restore_dict = torch.load(pretrained_wts_file,map_location=device)
        backbone.load_state_dict(restore_dict, strict=True)
    return backbone


# Define the full model
class SegmentationModel(nn.Module):
    def __init__(self, backbone, seg_head):
        super().__init__()
        self.backbone = backbone
        self.seg_head = seg_head

    def forward(self, x):
        w = torch.nn.functional.interpolate(x[1], size=(96, 160), mode='nearest')
        features = self.backbone(x[0])
        z = (features[0]*w)+features[0]
        seg_map = self.seg_head(z)
        return seg_map

def run(args,dataloader,dataloader_val):
        
    if args.gpu==True:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Define the segmentation head
    num_classes = 2 # number of classes in your segmentation task
    if args.type_output=='diff':
        num_classes = 4
        
    seg_head = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3,padding='same'),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=3,padding='same'),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
    )

    backbone = loadbackbone(device)
    model = SegmentationModel(backbone, seg_head).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs=args.epoch;
    
    best_val_loss=10000;
    # Train the model
    import sys
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader);i=0;
        for images, weak, masks in pbar:
            optimizer.zero_grad()
            outputs = model([images.to(device),weak.to(device)])
            masks = masks.long().to(device)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            """
            with torch.no_grad():
              outputs = model([images.to(device),weak.to(device)])
              y = outputs.detach().cpu().numpy();g = np.argmax(y[0], axis=0)
              str_num = '{:05d}'.format(epoch)
              cv2.imwrite('/content/out/'+str(str_num)+'.png',addtext(g,'epoch-'+str(epoch)))
            # Update the progress bar with the current loss value
            """
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
            
            if (i+1)%args.saveiter==0:

              #validation ##################################
              print('validation...>')

              # Evaluate the model on validation set
              model.eval()
              val_loss = 0
              with torch.no_grad():
                  #pbar_val = tqdm(dataloader_val)
                  for images_val, weak_val, masks_val in dataloader_val:
                      outputs_val = model([images_val.to(device),weak_val.to(device)])
                      masks_val = masks_val.long().to(device)
                      tmp = criterion(outputs_val, masks_val).item()
                      val_loss += tmp
                      #sys.stdout.write("\033[K") # Clear to the end of line
                      #pbar_val.set_description('iter: '+str(i)+f" Epoch {epoch+1}/{num_epochs}, VAL-Loss: {tmp:.4f}")
                      #print('iter: '+str(i)+f" Epoch {epoch+1}/{num_epochs}, VAL-Loss: {tmp:.4f}")
                      #print("\033[K")
                      #sys.stdout.write('\r')

              val_loss /= len(dataloader_val)
              print('iter: '+str(i)+ f" ======>>> Epoch {epoch+1}/{num_epochs}, Mean VAL-Loss: {val_loss:.4f}")
              # Save the best model based on validation loss
              #print('')
              if val_loss < best_val_loss:
                  best_val_loss = val_loss
                  #os.system('rm '+args.model_dir)
                  torch.save(model.state_dict(), args.model_dir)

              model.train()
              #print('##################################################')
            i=i+1;
