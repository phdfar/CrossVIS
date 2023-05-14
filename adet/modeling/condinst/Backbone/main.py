from argparse import ArgumentParser
import run
import numpy as np
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def main(args): 
  run.start(args)
  return args 

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str ,default='train', required=False)
    parser.add_argument('--task', type=str ,default='correct', required=False)
    parser.add_argument('--network', type=str ,default='simple_backbone', required=False)
    parser.add_argument('--model_dir', type=str , required=False)
    parser.add_argument('--imagesize', type=tuple_type, required=True)
    parser.add_argument('--batchsize', type=int ,default=32, required=False)
    parser.add_argument('--epoch', type=int ,default=30, required=False)
    parser.add_argument('--basepath', type=str , default='/content/', required=False)
    parser.add_argument('--rgbpath', type=str , default='/content/', required=False)
    parser.add_argument('--otherpath', type=str , default='/content/', required=False)
    parser.add_argument('--loss', type=str , default='BCE', required=False)
    parser.add_argument('--restore', type=bool , default=False, required=False)
    parser.add_argument('--gpu', type=bool , default=False, required=False)
    parser.add_argument('--saveiter', type=int , default=3000, required=False)
    parser.add_argument('--type_output', type=str , default='mask', required=False)

    args = parser.parse_args()

    main(args)
