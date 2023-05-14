import random
from dataloader import correct
from net import simple_backbone

def start(args):
    if args.task=='correct':
        dataloader_train,dataloader_val=correct.run(args)
        if args.network=='simple_backbone':
            simple_backbone.run(args,dataloader_train,dataloader_val)
        
        

  