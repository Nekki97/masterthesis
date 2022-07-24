import traceback
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

np.random.seed(1)
tf.random.set_seed(1)

assert cfg.n_files >= 3, "Not enough files to load and split into train/val/test"

cfg.output_root += cfg.trial_name + "/"
Path(cfg.output_root).mkdir(parents=True, exist_ok=True)

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

for trialID in list(cfg.loop_settings.keys()):
	model_name = cfg.loop_settings[trialID]["model_name"]
	loss_weights_ID = cfg.loop_settings[trialID]["loss_weights_ID"]
	while True:
		try:

			# Configure current for loop model settings
			cfg.trial_settings = {"modelname"			: model_name,
							  	  "steps_per_epoch"		: 0,
							  	  "training_img_shapes"	: [],
							  	  "loss_weights"		: loss_weights_ID}
			
			tf.keras.backend.clear_session()

			################################################# GET DATA #############################################
			with tf.device('/CPU:0'):
				train_datasets = []
				val_datasets = []
				val_labels_sets = []
				test_datasets = []

				np_train_datasets = []
				np_val_datasets = []
				np_test_datasets = []

				for dataset_name in [cfg.xcat_dataset_name, cfg.patient_dataset_name]:
					if cfg.verbose: print("\n== Loading %s dataset ==\n"%(dataset_name))
					if not check_if_preprocessed(dataset_name) or cfg.preprocess:
						# Data/Labels in shape (files, img_dim, img_dim, n_imgs)
						data, labels = load_data(dataset_name, n_data = cfg.n_files)
						data, labels = preprocess(data, labels, size = cfg.image_shape[0])

						if cfg.visualize_data:
							show_np_data(data, labels, preprocessed=False)

						# Dataset in shape (channel, img_dim, img_dim, n_imgs)
						# channel 1: data
						# (channel 2: labels) if labels exist
						dataset = combine_data(data, labels)

						split_into_groups_and_save(dataset_name, dataset)
					else:
						if cfg.verbose: print("Found preprocessed data!\n")

					if cfg.verbose: print("Loading preprocessed data!\n")
					load_path = os.path.join(cfg.local_data_root, "preprocessed_data", dataset_name+"%s-%d-preprocessed.npz"%(str(cfg.image_shape), cfg.n_files))
					with np.load(load_path) as data:
						train_data_temp = data['train_data']
						val_data_temp 	= data['val_data']
						test_data 		= data['test_data']


					# Limit training data for various time-saving reasons
					if cfg.image_count > 0:
						train_data_temp = train_data_temp[:cfg.image_count,:,:,:]

					train_data 	= np.expand_dims(train_data_temp[:,:,:,0],axis=3)
					train_label = np.expand_dims(train_data_temp[:,:,:,1],axis=3)
					val_data 	= np.expand_dims(val_data_temp[:,:,:,0]	,axis=3)
					val_label 	= np.expand_dims(val_data_temp[:,:,:,1]	,axis=3)

					np_train_datasets.append(train_data)
					np_val_datasets.append(val_data)
					np_test_datasets.append(test_data)

					if cfg.visualize_data:
						show_np_data(train_data, train_label, preprocessed=True)

					train_dataset 	= tf.data.Dataset.from_tensor_slices(train_data)
					val_dataset 	= tf.data.Dataset.from_tensor_slices(val_data)
					val_labels 		= tf.data.Dataset.from_tensor_slices(val_label)
					test_dataset 	= tf.data.Dataset.from_tensor_slices(test_data)

					train_datasets.append(train_dataset)
					val_datasets.append(val_dataset)
					val_labels_sets.append(val_labels)
					test_datasets.append(test_dataset)

				# Define folder coding
				coding = "G%02d-F%02d-M%02d-L%02d-I%03d-C%03d-E%03d" % (cfg.all_datasets[cfg.xcat_dataset_name]["ID"], 
																	    cfg.all_datasets[cfg.patient_dataset_name]["ID"], \
																        cfg.models_cfg[model_name]["ID"], 
																        cfg.loss_weight_cfg[cfg.trial_settings["loss_weights"]]["ID"], 
																        cfg.image_shape[0], 
																        cfg.image_count,
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


	  	  		#tf.random.set_seed(1)
				batched_val_datasets = [val_datasets[0].batch(cfg.batch_size), \
									    val_datasets[1].batch(cfg.batch_size)]

	    		#tf.random.set_seed(1)
				batched_val_labels = [val_labels_sets[0].batch(cfg.batch_size), \
								      val_labels_sets[1].batch(cfg.batch_size)]


				# Call model according to settings
				model = get_model(training=True)

				# Define Callback
				checkpoint_filepath = get_available_checkpoint_path(coding)
				model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath+"/epoch_{epoch:03d}/checkpoint_file_{epoch:03d}", verbose=1, save_weights_only=True, save_freq=int(cfg.save_period*cfg.trial_settings["steps_per_epoch"]))

				plotter = NEWGANMonitor(batched_val_datasets, batched_val_labels, checkpoint_filepath, num_img=8, period=cfg.save_period)

				keras.callbacks.ProgbarLogger()
				
				Path(os.path.join(cfg.output_root,"logs")).mkdir(parents=True, exist_ok=True)
				# Access with "tensorboard --logdir=logs" in cmd (path here is just "logs")
				tensorboard = keras.callbacks.TensorBoard(log_dir=os.path.join(cfg.output_root,"logs"))

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

		except Exception:
			traceback.print_exc()
			continue
		break