"""
Transforms mp4 audio to npz. Code has strong assumptions on the dataset organization!

"""

import errno
import os
import glob
import librosa
import numpy as np

basedir = r'/mnt/hdd2/alvanaki/raw_data'
basedir_to_save = r'/mnt/hdd2/alvanaki/audio_dataset'
filenames = glob.glob(os.path.join(basedir, '*', '*', '*.mp4'))
for filename in filenames:
    data = librosa.load(filename, sr=16000)[0][-19456:]
    path_to_save = os.path.join(basedir_to_save,
                                filename.split('/')[-3],
                                filename.split('/')[-2],
                                filename.split('/')[-1][:-4]+'.npz')
    print(path_to_save)
    if not os.path.exists(os.path.dirname(path_to_save)):
        try:
            os.makedirs(os.path.dirname(path_to_save))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    np.savez( path_to_save, data=data)

##################----------limited size dataset code follows------------------###################

# filenames = glob.glob(os.path.join(basedir, '*', '*', '*.mp4'))
#
# folders=glob.glob(os.path.join(basedir, '*'))
# for folder in folders:
#     filenames = glob.glob(os.path.join(folder, '*', '*.mp4'))
#     count=0
#     val_count=0
#     test_count=0
#
#     for filename in filenames:
#         if "train" in filename and count<=60:
#             count+=1
#
#         elif "train" in filename and count>60:
#
#             continue
#         elif "test" in filename and  test_count<=10:
#             test_count+=1
#         elif "test" in filename and  test_count>10:
#             continue
#         if "val" in filename and val_count<=10:
#             val_count+=1
#         elif "val" in filename and val_count>10:
#             continue
#
#
#         # for filename in filenames:
#         data = librosa.load(filename, sr=16000)[0][-19456:]
#         path_to_save = os.path.join(basedir_to_save,
#                                     filename.split('/')[-3],
#                                     filename.split('/')[-2],
#                                     filename.split('/')[-1][:-4]+'.npz')
#         if not os.path.exists(os.path.dirname(path_to_save)):
#             try:
#                 os.makedirs(os.path.dirname(path_to_save))
#             except OSError as exc:
#                 if exc.errno != errno.EEXIST:
#                     raise
#         np.savez( path_to_save, data=data)