from data_loader import *
from main_helper import *
# from data_loader import *
from cyclegan import *
# from main_helper import *
import config as cfg
from generators import *
from discriminators import *

# CUDA 11.2, cuDNN 8.4.1
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


test


np.random.seed(1)
tf.random.set_seed(1)

cfg.output_root += cfg.trial_name + "/"
Path(cfg.output_root).mkdir(parents=True, exist_ok=True)

for dataset_pair_names in cfg.dataset_pairs:
	for model_name 		in cfg.models:
		for loss_weights_number in cfg.loss_weights:
			while True:
				try:
					physical_devices = tf.config.list_physical_devices('GPU') 
					tf.config.experimental.set_memory_growth(physical_devices[0], True)


					# Configure current for loop model settings
					cfg.trial_settings = {"modelname"			: model_name,
									  	  "steps_per_epoch"		: 0,
									  	  "training_img_shapes"	: [],
									  	  "loss_weights"		: loss_weights_number}

									  	  # "g_lossID"			: cfg.generator_loss_options[generator_loss]["ID"],
									  	  # "g_loss"				: generator_loss,

					
					tf.keras.backend.clear_session()

					################################################# GET DATA #############################################
					with tf.device('/CPU:0'):
						train_datasets = []
						val_datasets = []
						test_datasets = []

						np_train_datasets = []
						np_val_datasets = []
						np_test_datasets = []

						for dataset_name in dataset_pair_names:
							if cfg.verbose: print("\n== Loading %s dataset ==\n"%(dataset_name))
							if not check_if_preprocessed(dataset_name) or cfg.preprocess:
								# Data/Labels in shape (files, img_dim, img_dim, n_imgs)
								data, labels = load_data(dataset_name, n_data = 3)
								data, labels = preprocess(data, labels, size = cfg.image_shape[0])

								# show_np_data(data, labels)

								# Dataset in shape (channel, img_dim, img_dim, n_imgs)
								# channel 1: data
								# (channel 2: labels) if labels exist
								dataset = combine_data(data, labels)

								split_into_groups_and_save(dataset_name, dataset)
							else:
								if cfg.verbose: print("Found preprocessed data!\n")

							load_path = os.path.join(cfg.local_data_root, "preprocessed_data", dataset_name+"-preprocessed.npz")
							with np.load(load_path) as data:
								train_data 		= data['train_data']
								val_data 		= data['val_data']
								test_data 		= data['test_data']


							# Limit training data for various time-saving reasons
							if cfg.image_count > 0:
								train_data = train_data[:cfg.image_count,:,:,:]

							np_train_datasets.append(train_data)
							np_val_datasets.append(val_data)
							np_test_datasets.append(test_data)

							# show_np_data(train_data, train_labels)

							train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
							val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
							test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

							train_datasets.append(train_dataset)
							val_datasets.append(val_dataset)
							test_datasets.append(test_dataset)

						# Define folder coding
						coding = "G%02d-F%02d-M%02d-L%02d-I%03d-E%03d" % (cfg.all_datasets[dataset_pair_names[0]]["ID"], 
																		  cfg.all_datasets[dataset_pair_names[1]]["ID"], \
																	      cfg.models_cfg[model_name]["ID"], 
																	      cfg.loss_weight_cfg[loss_weights_number]["ID"], 
																	      cfg.image_shape[0], 
																	      cfg.n_epochs)

						# Needed for both training and testing!
						model_save_path = os.path.join(cfg.output_root,"final_models/",coding)
						Path(model_save_path).mkdir(parents=True, exist_ok=True)


					################################################# TRAINING #############################################
					if cfg.train:
						
						cfg.trial_settings["training_img_shapes"] = [np_train_datasets[0].shape, np_train_datasets[1].shape]
						cfg.trial_settings["steps_per_epoch"] = min(np_train_datasets[0].shape[0], np_train_datasets[1].shape[0]) // cfg.batch_size

						batched_train_datasets = [train_datasets[0].batch(cfg.batch_size), \
												  train_datasets[1].batch(cfg.batch_size)]

						batched_val_datasets = [val_datasets[0].batch(cfg.batch_size), \
											    val_datasets[1].batch(cfg.batch_size)]

						# Call model according to settings
						model = get_model(training=True)

						# Define Callback
						checkpoint_filepath = os.path.join(cfg.output_root,"model_checkpoints",coding)
						Path(checkpoint_filepath).mkdir(parents=True, exist_ok=True)
						model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath+"/epoch_{epoch:03d}/checkpoint_file_{epoch:03d}", verbose=1, save_weights_only=True, save_freq=int(cfg.save_period*cfg.trial_settings["steps_per_epoch"]))

						plotter = NEWGANMonitor(batched_val_datasets, checkpoint_filepath, num_img=8, period=cfg.save_period)

						keras.callbacks.ProgbarLogger()
						
						Path(os.path.join(cfg.output_root,"logs")).mkdir(parents=True, exist_ok=True)
						# Access with "tensorboard --logdir=logs" in cmd (path here is just "logs")
						tensorboard = keras.callbacks.TensorBoard(log_dir=os.path.join(cfg.code_root,"logs"))

						if cfg.verbose: print("Model input:\n", batched_train_datasets[0], "\n", batched_train_datasets[1], "\n")
						if cfg.verbose: print("TRAINING START!")

						with tf.device('/device:GPU:0'):
							history = model.fit(tf.data.Dataset.zip((batched_train_datasets[0], batched_train_datasets[1])),
							    	  			epochs=cfg.n_epochs,
										    	verbose=1,
										    	callbacks=[model_checkpoint_callback, 
										    			   plotter, 
										    			   tensorboard],
												validation_data=tf.data.Dataset.zip((batched_val_datasets[0], batched_val_datasets[1])),
												steps_per_epoch=cfg.trial_settings["steps_per_epoch"])

						with tf.device('/CPU:0'):
							if cfg.verbose: print("\nSaving history data!")
							np.save(checkpoint_filepath+"/history.npy", history.history)
							# load using history = np.load('history.npy', allow_pickle='TRUE').item()

							generate_loss_plots(model_save_path, history)

							if cfg.verbose: print("Saving model!")
							Path(model_save_path+"/model/").mkdir(parents=True, exist_ok=True)
							model.save(model_save_path+"/model/")

							if cfg.verbose: print("\nTraining finished!")



					################################################# TESTING ##############################################
					with tf.device('/CPU:0'):
						batched_test_datasets = [test_datasets[0].batch(cfg.batch_size), \
											     test_datasets[1].batch(cfg.batch_size)]
						if cfg.test:
							new_generate_data(model_save_path, batched_test_datasets, n_gen_imgs=1)

					########################################################################################################
					tf.keras.backend.clear_session()

				except:
					cuda.select_device(0)
					cuda.close()
					continue
				break