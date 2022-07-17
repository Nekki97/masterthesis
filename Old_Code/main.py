from data_loader import *
from model import *
from main_helper import *
import config as cfg

#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

# CUDA 11.2, cuDNN 8.4.1
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

np.random.seed(1)
tf.random.set_seed(1)

for dataset_pair_names in cfg.dataset_pairs:
	for model_name 		in cfg.models:
		for generator_loss in cfg.generator_losses:
			for aug_strategy in cfg.augmentation_strategies:

				# Configure current for loop model settings
				trial_settings = {"modelID"		: cfg.models_cfg[model_name]["ID"],
								  "modelname"	: model_name,
								  "g_lossID"	: cfg.generator_loss_options[generator_loss]["ID"],
								  "g_loss"		: generator_loss,
								  "datasetGID"	: cfg.all_datasets[dataset_pair_names[0]]["ID"],
								  "datasetFID"	: cfg.all_datasets[dataset_pair_names[1]]["ID"]}

				################################################# GET DATA #############################################
				train_datasets = []
				train_datasets_np = []
				val_datasets = []
				test_datasets = []
				for dataset_name in dataset_pair_names:
					train_ds, val_ds, test_ds = load_datasets(dataset_name)
					train_datasets.append(train_ds)
					val_datasets.append(val_ds)
					test_datasets.append(test_ds)

					# Augment dataset and turn to numpy array
					if cfg.augment:
						train_dataset_np = augment_data(train_ds, aug_strategy, cfg.verbose)
					else:
						train_dataset_np = make_dataset_numpy(train_ds, "", verbose=True)

					# Visualize some data
					if cfg.visualize_data:
						visualize_dataset(train_dataset_np, dataset_name)

				# Define folder coding
				coding = "G%02d-F%02d-M%02d-L%02d-I%03d-E%03d" % (trial_settings["datasetGID"], trial_settings["datasetFID"], \
					trial_settings["modelID"], trial_settings["g_lossID"], cfg.image_shape[0], cfg.n_epochs)

				# Needed for both training and testing!
				model_save_path = os.path.join(cfg.output_root,"final_models/",coding)
				Path(model_save_path).mkdir(parents=True, exist_ok=True)


				################################################# TRAINING #############################################
				if cfg.train:
					# Call model according to settings
					model = get_model(trial_settings, training=True)

					# Define Callback
					checkpoint_filepath = os.path.join(cfg.output_root,"model_checkpoints",coding)
					Path(checkpoint_filepath).mkdir(parents=True, exist_ok=True)
					model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath+"/epoch_{epoch:03d}", verbose=1, save_freq=int(cfg.save_period*cfg.steps_per_epoch))

					#predictions_path = os.path.join(cfg.output_root,"model_predictions/%02d-%02d/"%(trial_settings["nameID"], trial_settings["g_lossID"]))
					#Path(predictions_path).mkdir(parents=True, exist_ok=True)
					plotter = GANMonitor(val_datasets, checkpoint_filepath, num_img=8, period=cfg.save_period)

					keras.callbacks.ProgbarLogger()
					
					Path(os.path.join(cfg.code_root,"logs")).mkdir(parents=True, exist_ok=True)
					# Access with "tensorboard --logdir=logs" in cmd (path here is just "logs")
					tensorboard = keras.callbacks.TensorBoard(log_dir=os.path.join(cfg.code_root,"logs"))


					if cfg.verbose: print("TRAINING START!")
					with tf.device('/device:GPU:0'):
						history = model.fit(tf.data.Dataset.zip((train_datasets[0], train_datasets[1])),
						    	  			epochs=cfg.n_epochs,
									    	verbose=1,
									    	callbacks=[model_checkpoint_callback, 
									    			   plotter, 
									    			   tensorboard],
											validation_data=tf.data.Dataset.zip((val_datasets[0], val_datasets[1])),
											steps_per_epoch=cfg.steps_per_epoch)

					if cfg.verbose: print("\nSaving history data!")
					np.save(checkpoint_filepath+"/history.npy", history.history)
					# load using history = np.load('history.npy', allow_pickle='TRUE').item()

					generate_loss_plots(model_save_path, history)

					if cfg.verbose: print("Saving model!")
					model.save(model_save_path)

					if cfg.verbose: print("\nTraining finished!")



				################################################# TESTING ##############################################
				if cfg.test:
					generate_data(model_save_path, trial_settings, test_datasets, n_gen_imgs=4)
