import pydicom
#import matplotlib.pyplot as plt
import numpy as np
import os
import keras
from keras.utils import array_to_img
import tensorflow as tf
import sys
import SimpleITK as sitk
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from SegClassRegBasis import preprocessing, normalization, segbasisloader
from keras import layers

import config as cfg



class GANMonitor(keras.callbacks.Callback):
		"""A callback to generate and save images after each epoch"""

		def __init__(self, val_data=None, path="", num_img=4, period=1):
			self.num_img = num_img
			self.val_data = val_data
			self.code_root = cfg.code_root
			self.path = path
			self.period = period
			self.epochs_since_last_save = 0

		def on_epoch_end(self, epoch, logs=None):
			self.epochs_since_last_save += 1
			if self.epochs_since_last_save % self.period == 0:
				for k in range(2):
					fig, ax = plt.subplots(2, self.num_img, figsize=(3*self.num_img, 6))
					for i, img in enumerate(self.val_data[k].take(self.num_img)):
						
						if k==0: prediction = self.model.gen_G(img)[0].numpy()
						if k==1: prediction = self.model.gen_F(img)[0].numpy()

						prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
						img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

						ax[0, i].imshow(img, cmap='gray')
						ax[1, i].imshow(prediction, cmap='gray')
						ax[0, i].set_title("Input image")
						ax[1, i].set_title("Translated image")
						ax[0, i].axis("off")
						ax[1, i].axis("off")

					folder_path = os.path.join(self.path,"epoch_%03d"%(epoch+1))
					Path(folder_path+"/_generated_data").mkdir(parents=True, exist_ok=True)

					if k==0: fig.savefig(os.path.join(folder_path,"_generated_data",'genG_after_%03d-of-%03d_epochs.png'%(epoch+1, cfg.n_epochs)))
					if k==1: fig.savefig(os.path.join(folder_path,"_generated_data",'genF_after_%03d-of-%03d_epochs.png'%(epoch+1, cfg.n_epochs)))
					plt.close(fig)



def make_dataset_numpy(dataset, use, verbose=False):
	"""
    Turns the dataset into a numpy array

    Parameters
    ----------
    dataset: tf.data.Dataset
    	dataset used in processing.py functions
    use: string
    	what to write in the verbose comments e.g. "train"
	verbose : boolean
    	Step by step description in terminal
		
    Returns
    -------
    Array
    	numpy array of the dataset
    """
	dataset_np = list([])
	for elem in dataset:
		dataset_np.append(elem.numpy())
	dataset_np = np.array(dataset_np)
	if verbose: print("Numpy shape: \t\t%s" % (str(dataset_np.shape)))
	return dataset_np






def visualize_dataset(np_array, name):
	"""
    Visualizes max. 16 images of a dataset

    Parameters
    ----------
    np_array: numpy array
		Array to visualize data from
	name: str
		name of dataset to easier identify in plot
    """
	if cfg.verbose: print("\nShowing dataset sample (close plot window to continue)")
	np.random.shuffle(np_array)
	batch_size = np_array.shape[1]
	show_n_images = min(16, np_array.shape[0]*np_array.shape[1])

	_, ax = plt.subplots(4, 4, figsize=(10, 15))
	plt.suptitle(name)
	for i in range(show_n_images):
		image = np_array[i//batch_size,i%batch_size,:,:,0]
		ax[i//4,i%4].imshow(image)
	plt.show()





def augment_data(tf_dataset, strategy, verbose=False):
	"""
    Augments data by rotating, flipping, rescaling and/or by adding gaussian or poisson noise

    Parameters
    ----------
    tf_dataset: tf.data.Dataset
    	dataset used in processing.py functions
    augment_cfg: dictionary
    	dictionary containing detailed augmentation instructions
	verbose : boolean
    	Step by step description in terminal
		
    Returns
    -------
    Array
    	numpy array of the dataset and the augmented dataset
    """
	
	# Then convert into numpy array
	aug_cfg = cfg.augmentation_cfg[strategy]

	np_dataset = make_dataset_numpy(tf_dataset, "", verbose=True)
	test_img = np_dataset[0,0,:,:,0]

	# Then implement numpy augmentations
	np_aug = np_dataset
	if aug_cfg["add_noise"]:
		for batch in range(np_dataset.shape[0]):
			for image in range(np_dataset.shape[1]):
				if (aug_cfg["noise_type"] == "gaussian"):
					np_aug[batch,image,:,:,0] += np.random.normal(0, aug_cfg["standard_deviation"], test_img.shape)
				elif (aug_cfg["noise_type"] == "poisson"):
					np_aug[batch,image,:,:,0] += np.random.poisson(aug_cfg["mean_poisson"], test_img.shape) * -(aug_cfg["mean_poisson"])
		np_dataset = np.concatenate((np_dataset, np_aug),axis=0)
	if verbose: print("Augmented dataset to \t%s using strategy: %s!" %(str(np_dataset.shape), strategy))
	return np_dataset

	



def generate_data(model_save_path, trial_settings, test_datasets, n_gen_imgs=1):
	if cfg.verbose: print("\nGenerating data!\n")
	test_model = get_model(trial_settings, training=False)

	if len(os.listdir(model_save_path)) != 0:

		# Load the final model
		model = keras.models.load_model(model_save_path)
		if cfg.verbose: print("Weights loaded!\n")

		# Generate n_gen_imgs images for both
		Path(model_save_path+"/_generated_data").mkdir(parents=True, exist_ok=True)

		for k in range(2):
			gen_imgs = n_gen_imgs
			m = 0
			while gen_imgs > 0:
				if k==0: filename = 'final_prediction_genG_%03d_%03d.png'%(cfg.n_epochs, m)
				if k==1: filename = 'final_prediction_genF_%03d_%03d.png'%(cfg.n_epochs, m)
				
				if filename not in os.listdir(model_save_path+"/_generated_data"):	
					fig, ax = plt.subplots(2, 8, figsize=(24, 6))
					for i, img in enumerate(test_datasets[k].take(8)):
						if k==0: prediction = model.gen_G(img, training=False)[0].numpy()
						if k==1: prediction = model.gen_F(img, training=False)[0].numpy()

						prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
						img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

						ax[0, i].imshow(img, cmap='gray')
						ax[1, i].imshow(prediction, cmap='gray')
						ax[0, i].set_title("Input image")
						ax[0, i].set_title("Input image")
						ax[1, i].set_title("Translated image")
						ax[0, i].axis("off")
						ax[1, i].axis("off")

					Path(model_save_path+"/_generated_data").mkdir(parents=True, exist_ok=True)

					path = os.path.join(model_save_path,"_generated_data",filename)
					fig.savefig(path)#, bbox_inches='tight')
					if cfg.verbose: print("Generated data saved as: %s"%(filename))
					plt.close(fig)
					
					gen_imgs -= 1
					m += 1
			else:
				continue
	else:
		if cfg.verbose: print("No model to load!")



def old_generate_loss_plots(model_save_path, history):
	if cfg.verbose: print("Saving loss plots!")
	Path(model_save_path+"/_generated_plots").mkdir(parents=True, exist_ok=True)
	fig = plt.figure(figsize=(10,5))
	plt.xlabel("epochs")
	plt.xticks(np.arange(1,cfg.n_epochs+1, int(cfg.n_epochs+1//10)))
	plt.ylim(bottom=0)

	y_max = 0
	print(history.history.keys())
	for key in history.history.keys():
		
		loss_vals = history.history[key]
		plt.plot(np.arange(1,cfg.n_epochs+1),loss_vals, label=key)
		
		if np.max(loss_vals) > y_max:
			y_max = np.max(loss_vals)

	plt.ylim(top=y_max)
	plt.legend()

	path = os.path.join(model_save_path,"_generated_plots",'total_%s.png'%(key))
	plt.savefig(path)
	plt.close(fig)
	if cfg.verbose: 
		print("Generated plot saved as: %s"%(path))


def old_get_model(training):
	if training:
		if cfg.verbose:
			print("\n========= Training Parameters: =========")
			print("- model name: \t\t%s" 	% (cfg.trial_settings["modelname"]))
			print("- batch size: \t\t%d" 	% (cfg.batch_size))
			#print("- samples per volume: \t%d" 	% (cfg.samples_per_volume))
			print("- epochs: \t\t%d" 	% (cfg.n_epochs))
			print("- steps per epoch: \t%d" 	% (cfg.trial_settings["steps_per_epoch"]))
			print("- input shape 1: \t%s"% str(cfg.trial_settings["training_img_shapes"][0]))
			print("- input shape 2: \t%s"% str(cfg.trial_settings["training_img_shapes"][1]))
			print("========================================\n")

	if cfg.trial_settings["modelname"] == "VanillaCycleGAN":
		
		# TODO: implement model_settings that change the generator loss in the model.py file!!!
		model = CycleGan(generator_G 		= get_resnet_generator(name="generator_G"), 
					   	 generator_F 		= get_resnet_generator(name="generator_F"),
					     discriminator_X 	= get_discriminator(name="discriminator_X"), 
					     discriminator_Y 	= get_discriminator(name="discriminator_Y"))

		if training:
			model.compile(
			    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
			    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
			    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
			    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
			    gen_loss_fn=generator_loss_fn,
			    disc_loss_fn=discriminator_loss_fn)
	return model