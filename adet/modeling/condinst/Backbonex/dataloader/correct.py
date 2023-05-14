import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
import random
import pickle

# Define your custom dataset class
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths,weak_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.weak_paths = weak_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask from file
        #image = Image.open(self.image_paths[idx]).convert('RGB')
        #mask = Image.open(self.mask_paths[idx]).convert('L')
        image= cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        #mask= np.ceil(cv2.imread(self.mask_paths[idx],0)/255)
        mask= cv2.imread(self.mask_paths[idx],0).astype('float')
        weak= cv2.imread(self.weak_paths[idx],0).astype('float')

        #mask = abs(mask-weak)
        # Apply transformations to image and mask

        # Convert NumPy arrays to PIL images
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        weak = Image.fromarray(weak)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            weak = self.transform(weak)

        #output_tensor = torch.nn.functional.interpolate(weak, size=(96, 160), mode='nearest')
        return image.float(), weak.float(), mask.squeeze(0).float()

def run(args):
        
    if args.type_output=='mask':
        type_output='gt'
    if args.type_output=='diff':
        type_output='diff'
        
    with open(args.otherpath+'pair_sample/sample.obj', 'rb') as handle:
        sample_file = pickle.load(handle)

    # Define the transformation to apply to the image and mask
    transform = transforms.Compose([
        transforms.Resize((384, 640)),
        transforms.ToTensor()
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image_paths=[];weak_paths=[];mask_paths=[]
    random.Random(1337).shuffle(sample_file)
    
    ln = len(sample_file)
    b=int(ln*0.85)
    c=int(ln*0.01)
    d=int(ln*0.14)
    
    print('Number train samples '+str(b))
    print('Number valid samples '+str(c))
    print('Number test samples '+str(d))
    print('###################################')
    
    # Define the paths to your images and masks
    for i in range(0,b):
      image_paths.append(args.rgbpath+sample_file[i]['img'])
      mask_paths.append(args.otherpath+sample_file[i][type_output])
      weak_paths.append(args.otherpath+sample_file[i]['weak'])

    
    # Create the dataset and dataloader
    dataset = SegmentationDataset(image_paths, mask_paths, weak_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    
    
    image_paths=[];weak_paths=[];mask_paths=[]
    for i in range(b+1,b+1+c):
      image_paths.append(args.rgbpath+sample_file[i]['img'])
      mask_paths.append(args.otherpath+sample_file[i][type_output])
      weak_paths.append(args.otherpath+sample_file[i]['weak'])

    # Create the dataset and dataloader
    datasetv= SegmentationDataset(image_paths, mask_paths, weak_paths, transform=transform)
    dataloader_val = DataLoader(datasetv, batch_size=args.batchsize, shuffle=True)
    
    return dataloader,dataloader_val


  