"""
Transforms mp4 videos to npz and crop fixed mouth ROIs. Code has strong assumptions on the dataset organization!

"""
import errno
import os
import cv2
import numpy as np
import glob


def extract_opencv(filename1):
    video = []
    cap = cv2.VideoCapture(filename1)

    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    return video[...,::-1]


basedir = r'/mnt/hdd2/alvanaki/raw_data'
basedir_to_save = r'/mnt/hdd2/alvanaki/video_dataset'
folders=glob.glob(os.path.join(basedir))
filenames = glob.glob(os.path.join(basedir, '*', '*', '*.mp4'))
for filename in filenames:
    data = extract_opencv(filename)[:, 115:211, 79:175]
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

# ###################------------code for reduced dataset size------------###################
# basedir = 'D:\\end-to-end-lipreading-master\\raw_data'
# basedir_to_save = r'D:\end-to-end-lipreading-master\datasets\visual_data_reduc'
# folders=glob.glob(os.path.join(basedir, '*'))
# for folder in folders:
#     filenames = glob.glob(os.path.join(folder, '*', '*.mp4'))
#     count=0
#     val_count=0
#     test_count=0
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
#         print(count)
#         data = extract_opencv(filename)[:, 115:211, 79:175]
#         path_to_save = os.path.join(basedir_to_save,
#                                     filename.split('\\')[-3],
#                                     filename.split('\\')[-2],
#                                     filename.split('\\')[-1][:-4]+'.npz')
#         if not os.path.exists(os.path.dirname(path_to_save)):
#             try:
#                 os.makedirs(os.path.dirname(path_to_save))
#             except OSError as exc:
#                 if exc.errno != errno.EEXIST:
#                     raise
#         np.savez( path_to_save, data=data)
