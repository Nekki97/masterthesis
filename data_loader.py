import os
import SimpleITK as sitk
from pathlib import Path
import glob
import numpy as np
import keras
from matplotlib import pyplot as plt
import tensorflow as tf
from numba import cuda

import config as cfg

def load_data(dataset_name, n_data):
	# Return all SITK-images and the resp. labels as np arrays if the path+"_masks" folder exists
	# Assumes the data and labels are in the same order and e.g. the first ones in each folder correspond
	# Makes sure all files have the same amount of numbers
	# Pads the images with zeros until theyre all the same size

	path = cfg.all_datasets[dataset_name]["path"]

	if len(glob.glob(path+"/*.nii.gz")) > 0:
		filetype = ".nii.gz"
	elif len(glob.glob(path+"/*.nrrd")) > 0:
		filetype = ".nrrd"

	# Find files
	filenames = [f for f in glob.glob(path+"/*"+filetype)]
	if cfg.verbose: print("Found %d %s files!"%(len(filenames), filetype))

	# Find pot. labels
	has_labels = os.path.isdir(path+"_masks")
	if has_labels:
		if len(glob.glob(path+"_masks"+"/*.nii.gz")) > 0:
			label_filetype = ".nii.gz"
		elif len(glob.glob(path+"_masks"+"/*.nrrd")) > 0:
			label_filetype = ".nrrd"
		label_filenames = [f for f in glob.glob(path+"_masks"+"/*"+label_filetype)]
		if cfg.verbose: print("Found %d %s label files!"%(len(label_filenames), label_filetype))

	if n_data == 0:
		n_data = len(filenames)

	# Load data
	data = []
	labels = []
	for i, filename in enumerate(filenames):
		if i+1>n_data:
			continue

		image = sitk.ReadImage(filename)

		if has_labels:

			# TODO: write target spacing to which all images are normed

			label = sitk.ReadImage(label_filenames[i])
			label.SetSpacing(image.GetSpacing())
			label.SetOrigin(image.GetOrigin())
			label.SetDirection(image.GetDirection())

			label = np.moveaxis(sitk.GetArrayFromImage(label),0,-1)
			labels.append(label)

		image = np.moveaxis(sitk.GetArrayFromImage(image),0,-1)
		data.append(image)


	# Cut off n_imgs to be equal over all files
	min_n_imgs = np.min([data[i].shape[2] for i in range(len(data))])
	data = [data[i][:,:,:min_n_imgs] for i in range(len(data))]
	if has_labels:
		labels = [labels[i][:,:,:min_n_imgs] for i in range(len(labels))]


	# Make image size equal by adding zeros to all smaller than the largest
	max_width  = np.max([data[i].shape[0] for i in range(len(data))])
	max_height = np.max([data[i].shape[1] for i in range(len(data))])
	if has_labels:
		max_width  = max(max_width,  np.max([labels[i].shape[0] for i in range(len(data))]))
		max_height = max(max_height, np.max([labels[i].shape[1] for i in range(len(data))]))

	# Make the minimum pixel value the default, as it can be negative --> zero wouldnt be the smallest
	min_pixel_val = min([np.min(data[i]) for i in range(len(data))])
	new_data = np.ones((len(data), max_width, max_height, min_n_imgs))*min_pixel_val
	new_labels = []

	if has_labels:
		min_pixel_val = min([np.min(labels[i]) for i in range(len(data))])
		new_labels = np.ones((len(data), max_width, max_height, min_n_imgs))*min_pixel_val
	for ID in range(len(data)):
		img = data[ID]
		if has_labels:
			label = labels[ID]

		new_data[ID, :img.shape[0],:img.shape[1],:] = img
		if has_labels:
			new_labels[ID, :label.shape[0],:label.shape[1],:] = label



	if cfg.verbose: print("Loaded %d files!\n%s\n"%(n_data,str(new_data.shape)))
	return new_data, new_labels


def preprocess(data, labels=[], size=0):
	# Clip at 5th and 95th percentile 
	# Normalize to [0,1]
	# Crop rows/columns with sum = 0.0
	# Pad image to be square and uniform across dataset
	# OR Pad to set size

	# If you want to view some sample images before and after preprocessing
	_view_sample_img = False
	_view_img_number = 20

	# Min amount of images per file
	_min_img_count = 1000 # ignore

	_has_labels = len(labels)>0

	# get max width where sum of pixels != 0.0
	max_width = np.max(np.sum(np.sum(data,axis=1) != 0.0,axis=1))
	
	# get max height where sum of pixels != 0.0
	max_height = np.max(np.sum(np.sum(data,axis=2) != 0.0,axis=1))

	if size == 0:
		pad_to_size = max(max_width, max_height)
	else:
		pad_to_size = size


	processed_files = []
	processed_label_files = []
	for fileID in range(data.shape[0]):
		processed_images = []
		processed_labels = []
		for imgID in range(data.shape[-1]):
			image = data[fileID,:,:,imgID]
			if _has_labels:
				label = labels[fileID,:,:,imgID]

			# Skip image if empty
			if np.sum(image) == 0.0:
				continue

			# Threshold labels and skip if empty
			if _has_labels:
				label[label>=0.5] = 1.0
				label[label<0.5] = 0.0
				if np.sum(label) == 0.0:
					continue

			# View sample image before preprocessing
			if _view_sample_img and imgID == _view_img_number:
				plt.subplot(2,2,1)
				plt.imshow(image, cmap="gray")
				if _has_labels:
					plt.subplot(2,2,2)
					plt.imshow(label, cmap="gray")


			# Clip image
			image = np.clip(image, np.percentile(image, 5), np.percentile(image, 95))


			# Skip image if empty (might happen after clipping)
			if np.sum(image) == 0.0:
				continue

			# Normalize image (if padding with nans instead of zeros use nanmin and nanmax)
			# Normalize to [0,1]
			if cfg.normalization_cfg[cfg.normalization_ID][0] == 0:
				image = (image-np.min(image))/(np.max(image)-np.min(image))
				pad_val = 0 # new min value to pad with 
			
			# Normalize to [-1,1]
			elif cfg.normalization_cfg[cfg.normalization_ID][0] == -1:
				image = 2*np.ones_like(image)*(image-np.min(image))/(np.max(image)-np.min(image))-np.ones_like(image)
				pad_val = -1 # new min value to pad with


			# Crop image (if padding with nans instead of zeros use nansum)
			dim_0_mask = (np.sum(image,axis=0) != 0.0)
			dim_1_mask = (np.sum(image,axis=1) != 0.0)

			temp_image = image[:,dim_0_mask]
			if _has_labels:
				label = label[:,dim_0_mask]

			image = temp_image[dim_1_mask,:]
			if _has_labels:
				label = label[dim_1_mask,:]

			# Pad to be square
			diff = np.abs(image.shape[0]-image.shape[1])
			pad_dim = image.shape[0] > image.shape[1]

			image = np.pad(image, (((pad_dim==0)*(diff//2+diff%2), (pad_dim==0)*diff//2),((pad_dim==1)*(diff//2+diff%2), (pad_dim==1)*diff//2)), mode="constant", constant_values=pad_val)
			if _has_labels:
				label = np.pad(label, (((pad_dim==0)*(diff//2+diff%2), (pad_dim==0)*diff//2),((pad_dim==1)*(diff//2+diff%2), (pad_dim==1)*diff//2)), mode="constant", constant_values=pad_val)


			# Pad to common size for dataset by padding or cropping
			diff = pad_to_size-image.shape[0]
			if diff>0:
				image = np.pad(image, (((diff//2+diff%2), diff//2),((diff//2+diff%2), diff//2)), mode="constant", constant_values=pad_val)
				if _has_labels:
					label = np.pad(label, (((diff//2+diff%2), diff//2),((diff//2+diff%2), diff//2)), mode="constant", constant_values=pad_val)
			else:
				# If True --> randomly crop image
				if cfg.randomly_crop:
					original_img = image
					if _has_labels: 
						original_label = label

					start_1 = np.random.randint(-diff)
					start_2 = np.random.randint(-diff)
					image = original_img[start_1:start_1+pad_to_size,start_2:start_2+pad_to_size]
					if _has_labels:
						label = original_label[start_1:start_1+pad_to_size,start_2:start_2+pad_to_size]

					# If augment crop --> repeat random cropping n_augment times
					if cfg.augment_crop:
						augmented_imgs = []
						if _has_labels:
							augmented_labels = []
						for i in range(cfg.n_aug_crops):
							start_1 = np.random.randint(-diff)
							start_2 = np.random.randint(-diff)
							image = original_img[start_1:start_1+pad_to_size,start_2:start_2+pad_to_size]
							augmented_imgs.append(image)
							if _has_labels:
								label = original_label[start_1:start_1+pad_to_size,start_2:start_2+pad_to_size]
								augmented_labels.append(label)
				else:
					# Crop image in center
					center = image.shape[0]//2
					extra = image.shape[0]%2
					image = image[center-pad_to_size//2+pad_to_size%2 : center+extra+pad_to_size//2, center-pad_to_size//2+pad_to_size%2 : center+extra+pad_to_size//2]
					if _has_labels:
						label = label[center-pad_to_size//2+pad_to_size%2 : center+extra+pad_to_size//2, center-pad_to_size//2+pad_to_size%2 : center+extra+pad_to_size//2]


			# View sample image after preprocessing
			if _view_sample_img and imgID == _view_img_number:
				plt.subplot(2,2,3)
				plt.imshow(image, cmap="gray")
				if _has_labels:
					plt.subplot(2,2,4)
					plt.imshow(label, cmap="gray")
					plt.show()

			# Save all data in (file, data_dim, data_dim, n_imgs) structure
			processed_images.append(image)
			if _has_labels:
				processed_labels.append(label)

			if cfg.augment_crop:
				# Save augmented files too
				for aug_img in augmented_imgs:
					processed_images.append(aug_img)
				if _has_labels:
					for aug_lbl in augmented_labels:
						processed_labels.append(aug_lbl)

		# print([processed_images[i].shape for i in range(len(processed_images))])
		processed_images = np.array(processed_images)
		if _has_labels:
			processed_labels = np.array(processed_labels)

		if processed_images.shape[0] < _min_img_count:
			_min_img_count = processed_images.shape[0]

		processed_files.append(processed_images)
		if _has_labels:
			processed_label_files.append(processed_labels)

	# Make sure every file has the same amount of images
	processed_files = [processed_files[i][:_min_img_count,:,:] for i in range(len(processed_files))]
	if _has_labels:
		processed_label_files = [processed_label_files[i][:_min_img_count,:,:] for i in range(len(processed_label_files))]

	processed_data = np.moveaxis(np.array(processed_files),1,-1)
	if _has_labels:
		labels = np.moveaxis(np.array(processed_label_files),1,-1)

	if cfg.verbose: print("Finished preprocessing!\n%d files with %d images each of size %dx%d!\n%s\n" % (processed_data.shape[0], processed_data.shape[-1], pad_to_size, pad_to_size, str(processed_data.shape)))

	return processed_data, labels


def visualize_dataset(np_array):
	# Data in shape (file, image_dim, image_dim, image_slice)

	if cfg.verbose: print("\nShowing dataset sample (close plot window to continue)")
	np.random.shuffle(np_array)
	show_n_images = min(64, np_array.shape[1])

	_, ax = plt.subplots(8, 8, figsize=(15, 15))
	for i in range(show_n_images):
		image = np_array[i//np_array.shape[-1],:,:,i%np_array.shape[-1]]
		ax[i//8,i%8].imshow(image, cmap="gray")
	plt.show()


def check_if_preprocessed(dataset_name):
	# Checks if folder with preprocessed data exists for given dataset
	return os.path.exists(os.path.join(cfg.local_data_root, "preprocessed_data", dataset_name+"-preprocessed.npz"))
		


def combine_data(data, labels):
	# Combine data and labels or itself into one array with two channels

	data = np.expand_dims(data, axis=0)
	dataset = data

	if len(labels) > 0:
		labels = np.expand_dims(labels, axis=0)
	else:
		labels = np.copy(data)

	dataset = np.concatenate((data, labels), axis=0)

	if cfg.verbose: print("Combined data into %s\n"%str(dataset.shape))
	return dataset


def split_into_groups_and_save(dataset_name, dataset):
	# Split data into train/val/test images of each file (better for generalizability) 
	# Shuffle files and images to prevent e.g. test having only the lower half
	# Get all imgs of all files into one dim

	n_test_img = int(dataset.shape[-1]*cfg.test_frac)
	n_val_img = int(dataset.shape[-1]*cfg.val_frac)
	n_train_img = int(dataset.shape[-1] - n_test_img - n_val_img)

	fileIDs = np.arange(dataset.shape[1])
	np.random.shuffle(fileIDs)
	dataset = dataset[:,fileIDs,:,:,:]

	imgIDs = np.arange(dataset.shape[-1])
	np.random.shuffle(imgIDs)
	dataset = dataset[:,:,:,:,imgIDs]

	train_data = dataset[:,:,:,:,:n_train_img]
	val_data = dataset[:,:,:,:,n_train_img+1:n_train_img+n_val_img]
	test_data = dataset[:,:,:,:,n_train_img+n_val_img+1:]



	def _get_all_imgs(data):
		all_data = []	
		for fileID in range(data.shape[1]):
			all_data.append(data[:,fileID,:,:,:])
		return np.concatenate(all_data,axis=-1)

	train_data = _get_all_imgs(train_data)
	val_data = _get_all_imgs(val_data)
	test_data = _get_all_imgs(test_data)



	# The model needs data as (n_imgs, img_dim, img_dim, channels)
	train_data = np.moveaxis(np.moveaxis(train_data,0,-1), -2,0)
	val_data = np.moveaxis(np.moveaxis(val_data,0,-1), -2,0)
	test_data = np.moveaxis(np.moveaxis(test_data,0,-1), -2,0)

	path = os.path.join(cfg.local_data_root, "preprocessed_data/")
	Path(path).mkdir(parents=True, exist_ok=True)
	np.savez(path+dataset_name+"-preprocessed.npz", train_data=train_data, val_data=val_data, test_data=test_data)

	if cfg.verbose: print("Saved train/val/test data as %s"%path+dataset_name+"-preprocessed.npz")
	if cfg.verbose: print(train_data.shape, val_data.shape, test_data.shape)


