import os
import glob
import shutil
#f = glob.glob('/kaggle/input/patchloss13k/CrossVIS/output/CrossVIS_R_50_1x/model_*.pth')
f = glob.glob('output/CrossVIS_R_50_1x/model_*.pth')

print(f)
for x in f:
    name = x.split('_');name=name[-1];name = name.replace('.pth','.zip')
    print(name)
    os.system('python test_vis.py --config-file configs/CrossVIS/R_50_1x.yaml --json-file datasets/youtubevis/annotations/valid.json --opts MODEL.WEIGHTS '+x)
    os.system('zip crossvis_patch_16b_'+name+' results.json')
    os.system('rm results.json')
    #shutil.move('crossvis_640_'+name,'pth/')
    os.system('mv crossvis_patch_16b_'+name+' pth/')
