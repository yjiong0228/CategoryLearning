import glob
recording_dir = glob.glob('Recording/*')
recording_dir.sort(key=lambda x: (int(x.split('/')[-1].split('_')[0]), int(x.split('/')[-1].split('_')[1].split('.')[0])))

import os
remote_base_dir = 'gpu6/categoryLearning/'
for dir in recording_dir:
    remote_dir = remote_base_dir+dir
    files = glob.glob(dir+'/*')
    for file in files:
        os.system('onedrive-uploader upload '+file+' '+remote_dir)
