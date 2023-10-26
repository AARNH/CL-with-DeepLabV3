import os
import cv2
import csv
import pandas
import numpy as np
from tqdm import tqdm
import matplotlib.image as img
from skimage.draw import polygon
from collections import defaultdict

root = "./../dataset_echonet"  # Path data
save_path = "./../data" # Path untuk menyimpah hasil ekstraksi gambar
extract_all_images_between = True

for split in ['train','val','test']:

	files = []
	images = defaultdict(list)
	seg = defaultdict(lambda: defaultdict(list))

	os.makedirs(os.path.join(save_path, "echonet", "segmentation", 'imgs', split), exist_ok=True)
	os.makedirs(os.path.join(save_path, "echonet", "segmentation", 'masks', split), exist_ok=True)

	with open(os.path.join(root, "FileList.csv")) as f:
			reader = csv.DictReader(f, delimiter=',')
			for row in reader:
				if row['Split'] == split.upper():
					row['FileName'] += '.avi' if '.avi' not in row['FileName'] else row['FileName']
					files.append(row['FileName'])

	with open(os.path.join(root, "VolumeTracings.csv")) as f:
			reader = csv.DictReader(f, delimiter=',')
			for row in reader:
				name = row['FileName']
				if name in files:
					coords = tuple([float(row[c]) for c in ['X1','Y1','X2','Y2']])
					frame = int(row['Frame'])
					if frame not in seg[name]:
						images[name].append(frame)
					seg[name][frame].append(coords)

	for name in images:
		for f in images[name]:
			seg[name][f] = np.array(seg[name][f])

	for name in tqdm(files):

		if len(images[name]) >= 2:

			f1, f2 = images[name][0], images[name][-1]
			t1, t2 = seg[name][f1], seg[name][f2]

			cap = cv2.VideoCapture(os.path.join(root, "Videos", name))
			width, height, num_frames = [int(cap.get(x)) for x in [3,4,7]]
			seq = np.zeros((num_frames, height, width, 3), np.uint8)

			for i in range(num_frames):
				_, frame = cap.read()
				seq[i, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# Extract semua gambar pretraining
			if  extract_all_images_between:
				# Ekstraksi hanya 2 gambar untuk pretraining (frame f1 dan f2)
				os.makedirs(os.path.join(save_path, "echonet", "pretraining", split), exist_ok=True)
				if f1 < f2:
					img.imsave(os.path.join(save_path, "echonet", "pretraining", split.lower(), 
								name.split('.')[0] + '_' + str(f1) + '.png'), seq[f1, :, :])
					img.imsave(os.path.join(save_path, "echonet", "pretraining", split.lower(),
								name.split('.')[0] + '_' + str(f2) + '.png'), seq[f2, :, :])
				else:
					img.imsave(os.path.join(save_path, "echonet", "pretraining", split.lower(),
								name.split('.')[0] + '_' + str(f2) + '.png'), seq[f2, :, :])
					img.imsave(os.path.join(save_path, "echonet", "pretraining", split.lower(), 
								name.split('.')[0] + '_' + str(f1) + '.png'), seq[f1, :, :])

				# Hapus gambar-gambar lain yang mungkin ada dalam folder pretraining
				files_to_keep = [str(f1) + '.png', str(f2) + '.png']
				pretraining_folder = os.path.join(save_path, "echonet", "pretraining", split.lower())

				for filename in os.listdir(pretraining_folder):
					if filename not in files_to_keep:
						file_path = os.path.join(pretraining_folder, filename)
						os.remove(file_path)


			x1, y1, x2, y2 = t1[:, 0], t1[:, 1], t1[:, 2], t1[:, 3]
			x = np.rint(np.concatenate((x1[1:], np.flip(x2[1:])))).astype(np.int32)
			y = np.rint(np.concatenate((y1[1:], np.flip(y2[1:])))).astype(np.int32)

			r, c = polygon(y, x, (112, 112))
			gt = np.zeros((112, 112), np.float32)
			gt[r, c] = 1
			img.imsave(os.path.join(save_path, "echonet", "segmentation", "masks", split.lower(),
						name.split('.')[0] + '_' + str(f1) + '_0.png'), gt)

			x1, y1, x2, y2 = t2[:, 0], t2[:, 1], t2[:, 2], t2[:, 3]
			x = np.rint(np.concatenate((x1[1:], np.flip(x2[1:])))).astype(np.int32)
			y = np.rint(np.concatenate((y1[1:], np.flip(y2[1:])))).astype(np.int32)

			r, c = polygon(y, x, (112, 112))
			gt = np.zeros((112, 112), np.float32)
			gt[r, c] = 1
			img.imsave(os.path.join(save_path, "echonet", "segmentation", "masks", split.lower(),
						name.split('.')[0] + '_' + str(f2) + '_1.png'), gt)
			
			img.imsave(os.path.join(save_path, "echonet", "segmentation", "imgs", split.lower(),
						name.split('.')[0] + '_' + str(f1) + '_0.png'), seq[f1, :, :])
			img.imsave(os.path.join(save_path, "echonet", "segmentation", "imgs", split.lower(),
						name.split('.')[0] + '_' + str(f2) + '_1.png'), seq[f2, :, :])