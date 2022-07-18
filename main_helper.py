from data_loader import *
from cyclegan import *
from discriminators import *
from generators import *

def show_np_data(data, labels):
	if cfg.visualize_data:
		visualize_dataset(data)
		if len(labels)>0:
 			visualize_dataset(labels)


class NEWGANMonitor(keras.callbacks.Callback):
		"""A callback to generate and save images after each epoch"""

		def __init__(self, val_data=None, val_labels=None, path="", num_img=4, period=1):
			self.num_img = num_img
			self.val_data = val_data
			self.val_labels = val_labels
			self.code_root = cfg.code_root
			self.path = path
			self.period = period
			self.epochs_since_last_save = 0

		def on_epoch_end(self, epoch, logs=None):
			self.epochs_since_last_save += 1
			if self.epochs_since_last_save % self.period == 0:
				for k in range(2):
					fig, ax = plt.subplots(4, self.num_img, figsize=(6*self.num_img, 6*4))

					val_data_batch = self.val_data[k].take(self.num_img)
					val_labels_batch = self.val_labels[k].take(self.num_img)
					labels = list(enumerate(val_labels_batch))

					for i, img in enumerate(val_data_batch):
						j, label = labels[i]

						if k==0: 
							img_prediction = self.model.gen_G(img)[0].numpy()
							label_prediction = self.model.gen_G(label)[0].numpy()
						if k==1: 
							img_prediction = self.model.gen_F(img)[0].numpy()
							label_prediction = self.model.gen_F(label)[0].numpy()

						img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
						label = (label[0] * 127.5 + 127.5).numpy().astype(np.uint8)

						img_prediction = (img_prediction * 127.5 + 127.5).astype(np.uint8)
						label_prediction = (label_prediction * 127.5 + 127.5).astype(np.uint8)

						ax[0, i].imshow(img, cmap='gray')
						ax[0, i].set_title("Input image")
						ax[0, i].axis("off")
						
						if k==0: ax[1, i].imshow(label, cmap='gray')
						ax[1, i].set_title("Input label")
						ax[1, i].axis("off")

						ax[2, i].imshow(img_prediction, cmap='gray')
						ax[2, i].set_title("Translated image")
						ax[2, i].axis("off")
						
						if k==0: ax[3, i].imshow(label, cmap='gray')
						ax[3, i].set_title("Translated label")
						ax[3, i].axis("off")

						# thrsh_label_pred = np.copy(label_prediction)
						# thrsh_label_pred[thrsh_label_pred>0.5] = 1
						# thrsh_label_pred[thrsh_label_pred<=0.5] = 0
		
						# if k==0: ax[4, i].imshow(thrsh_label_pred, cmap='gray')
						# ax[4, i].set_title("Thresholded translated label")
						# ax[4, i].axis("off")

					folder_path = os.path.join(self.path,"epoch_%03d"%(epoch+1))
					Path(folder_path+"/_generated_data").mkdir(parents=True, exist_ok=True)

					if k==0: 
						fig.savefig(os.path.join(folder_path,"_generated_data",'genG_after_%03d-of-%03d_epochs.png'%(epoch+1, cfg.n_epochs)))
						np.savez(os.path.join(folder_path,"_generated_data",'genG_after_%03d-of-%03d_epochs.npz'%(epoch+1, cfg.n_epochs)), \
							input_img=img, input_label=label, output_img=img_prediction, output_label=label_prediction)#, thresh_out_label=thrsh_label_pred)

					if k==1: 
						fig.savefig(os.path.join(folder_path,"_generated_data",'genF_after_%03d-of-%03d_epochs.png'%(epoch+1, cfg.n_epochs)))
						np.savez(os.path.join(folder_path,"_generated_data",'genF_after_%03d-of-%03d_epochs.npz'%(epoch+1, cfg.n_epochs)), \
							input_img=img, input_label=label, output_img=img_prediction, output_label=label_prediction)#, thresh_out_label=thrsh_label_pred)
					plt.close(fig)


def new_generate_data(model_save_path, test_datasets, n_gen_imgs=1):
	if cfg.verbose: print("\nGenerating data!\n")
	test_model = get_model(training=False)

	if len(os.listdir(model_save_path)) != 0:

		# Load the final model
		model = keras.models.load_model(model_save_path+"/model/")
		if cfg.verbose: print("Weights loaded!\n")

		# Generate n_gen_imgs images for both
		Path(model_save_path+"/_generated_data").mkdir(parents=True, exist_ok=True)

		for k in range(2):
			cols = 32
			sets = 2 # set of outputs

			rows_per_set = 6

			gen_imgs = n_gen_imgs
			m = 0
			while gen_imgs > 0:
				if k==0: filename = 'final_prediction_genG_%03d_%03d.png'%(cfg.n_epochs, m)
				if k==1: filename = 'final_prediction_genF_%03d_%03d.png'%(cfg.n_epochs, m)
				
				if filename not in os.listdir(model_save_path+"/_generated_data"):	
					fig, ax = plt.subplots(rows_per_set*sets, cols, figsize=(cols*4, sets*rows_per_set*4))
					test_batch = test_datasets[k].take(sets*cols)
					for i, data in enumerate(test_batch):

						img 	= np.expand_dims(data[:,:,:,0],axis=3)
						label 	= np.expand_dims(data[:,:,:,1],axis=3)

						if k==0: 
							img_prediction = model.gen_G(img, training=False)[0].numpy()
							label_prediction = model.gen_G(label, training=False)[0].numpy()
						if k==1: 
							img_prediction = model.gen_F(img, training=False)[0].numpy()
							label_prediction = model.gen_F(label, training=False)[0].numpy()

						img 	= (img[0] 	* 127.5 + 127.5).astype(np.uint8)
						label 	= (label[0] * 127.5 + 127.5).astype(np.uint8)

						img_prediction 		= (img_prediction 	* 127.5 + 127.5).astype(np.uint8)
						label_prediction 	= (label_prediction * 127.5 + 127.5).astype(np.uint8)						

						ax[0+rows_per_set*(i//cols), i%cols].imshow(img, cmap='gray')
						ax[0+rows_per_set*(i//cols), i%cols].set_title("Input image")	
						ax[0+rows_per_set*(i//cols), i%cols].axis("off")

						if k==0: ax[1+rows_per_set*(i//cols), i%cols].imshow(label, cmap='gray')
						ax[1+rows_per_set*(i//cols), i%cols].set_title("Input label")
						ax[1+rows_per_set*(i//cols), i%cols].axis("off")

						ax[2+rows_per_set*(i//cols), i%cols].axis("off")

						ax[3+rows_per_set*(i//cols), i%cols].imshow(img_prediction, cmap='gray')
						ax[3+rows_per_set*(i//cols), i%cols].set_title("Translated image")
						ax[3+rows_per_set*(i//cols), i%cols].axis("off")

						if k==0: ax[4+rows_per_set*(i//cols), i%cols].imshow(label, cmap='gray')
						ax[4+rows_per_set*(i//cols), i%cols].set_title("Label")
						ax[4+rows_per_set*(i//cols), i%cols].axis("off")

						ax[5+rows_per_set*(i//cols), i%cols].axis("off")

						# thrsh_label_pred = np.copy(label_prediction)
						# thrsh_label_pred[thrsh_label_pred>0.5] = 1
						# thrsh_label_pred[thrsh_label_pred<=0.5] = 0

						# if k==0: ax[5+rows_per_set*(i//cols), i%cols].imshow(thrsh_label_pred, cmap='gray')
						# ax[5+rows_per_set*(i//cols), i%cols].set_title("Thresholded translated label")
						# ax[5+rows_per_set*(i//cols), i%cols].axis("off")


					Path(model_save_path+"/_generated_data").mkdir(parents=True, exist_ok=True)

					path = os.path.join(model_save_path,"_generated_data",filename)
					fig.savefig(path)#, bbox_inches='tight')
				
					np.savez(path[:-3]+'npz',input_img=img, output_img=img_prediction, input_label=label, output_label=label_prediction)#, thresh_out_label=thrsh_label_pred)
					
					if cfg.verbose: print("Generated data saved as: %s"%(filename))
					plt.close(fig)
					
					gen_imgs -= 1
				m += 1
			else:
				continue
	else:
		if cfg.verbose: print("No model to load!")




def generate_loss_plots(model_save_path, history):
	if cfg.verbose: print("Saving loss plots!")
	Path(model_save_path+"/_generated_plots").mkdir(parents=True, exist_ok=True)
	fig = plt.figure(figsize=(10,5))
	plt.xlabel("epochs")
	plt.xticks(np.arange(0,cfg.n_epochs+1, max(1,int(cfg.n_epochs//10))))
	plt.ylim(bottom=0)

	y_max = 0
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


def get_model(training):
	if training:
		if cfg.verbose:
			print("\n========= Training Parameters: =========")
			print("- model name: \t\t%s" 	% (cfg.trial_settings["modelname"]))
			print("- batch size: \t\t%d" 	% (cfg.batch_size))
			#print("- samples per volume: \t%d" 	% (cfg.samples_per_volume))
			print("- epochs: \t\t%d" 	% (cfg.n_epochs))
			print("- steps per epoch: \t%d" 	% (cfg.trial_settings["steps_per_epoch"]))
			print("- loss weights: %.1f, %.1f, %.1f, %.1f"%(cfg.loss_weight_cfg[cfg.trial_settings["loss_weights"]]["cycle"],\
															cfg.loss_weight_cfg[cfg.trial_settings["loss_weights"]]["identity"], \
															cfg.loss_weight_cfg[cfg.trial_settings["loss_weights"]]["gradient"],\
															cfg.loss_weight_cfg[cfg.trial_settings["loss_weights"]]["perception"]))
			print("- input shape 1: \t%s"% str(cfg.trial_settings["training_img_shapes"][0]))
			print("- input shape 2: \t%s"% str(cfg.trial_settings["training_img_shapes"][1]))
			print("========================================\n")

	if cfg.trial_settings["modelname"] == "VanillaCycleGAN":
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
	elif cfg.trial_settings["modelname"] == "AdvancedCycleGAN":
		model = CycleGan(generator_G 		= unet(name="generator_G"), 
					   	 generator_F 		= unet(name="generator_F"),
					     discriminator_X 	= patchgan(name="discriminator_X"), 
					     discriminator_Y 	= patchgan(name="discriminator_Y"))

		if training:
			model.compile(
			    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
			    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
			    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
			    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
			    gen_loss_fn=generator_loss_fn,
			    disc_loss_fn=discriminator_loss_fn)
	elif cfg.trial_settings["modelname"] == "DominikCycleGAN":
		model = CycleGan(generator_G 		= get_resnet_generator(name="generator_G"), 
					   	 generator_F 		= get_resnet_generator(name="generator_F"),
					     discriminator_X 	= patchgan(name="discriminator_X"), 
					     discriminator_Y 	= patchgan(name="discriminator_Y"))

		if training:
			model.compile(
			    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
			    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
			    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
			    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
			    gen_loss_fn=generator_loss_fn,
			    disc_loss_fn=discriminator_loss_fn)
	return model