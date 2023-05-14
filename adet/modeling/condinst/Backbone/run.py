import run_train
import numpy

def start(args):
  if args.mode=='train' or args.mode=='test':
      run_train.start(args)