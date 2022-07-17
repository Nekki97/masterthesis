import pydicom
#import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import sys
import SimpleITK as sitk
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from SegClassRegBasis import preprocessing, normalization, segbasisloader, config as ckm_cfg
import config as cfg
from more_itertools import take

def get_filePaths(root, dataset, verbose = False):
	'''
	Reads out all filepaths and their patient/file IDs and 
	weighting and saves them in a filepath list and a metadata tuple list


	used in get_dict()

    Parameters
    ----------
    root : string
        string filepath to root folder of dataset
    amount : int
    	amount of patients to read out
    	0 = all
    dataset: dict
    	dataset dictionary containing modality, filetype, relative path and weighting
	verbose : boolean
    	Step by step description in terminal

    Returns
    -------
    list
        List with all filePaths
    list
    	List of tuples containing (patient ID, file ID) and optional weighting for e.g. ".IMA"-files
	'''
	if verbose: print("(1/5 Preprocessing) Getting filepaths")

	total_file_count = 0

	filePaths = []
	custom_meta = []

	if dataset["name"] in ["ckm t2 ima", "ckm t1 ima"]:
		patients = os.listdir(root)
		for patient in patients:
			patient_file_count = 0
			# Go through all folders while saving the name in metadata for identification
			for weighting in os.listdir(os.path.join(root, patient)):
				if dataset["modality"] in weighting and weighting == dataset["weighting"]:
					for file in os.listdir(os.path.join(root, patient, weighting)):	
						if file.endswith(".IMA") or file.endswith(".ima"):
							total_file_count += 1
							patient_file_count += 1
							#print(patient, weighting, file)
							filePath = os.path.join(root, patient, weighting, file)
							#print(filePath)
							file_id = file[12+int(int(patient)>9):16+int(int(patient)>9)]
							custom_meta.append((patient, file_id))
							#print(custom_meta)
							filePaths.append(filePath)

	elif dataset["name"] in ["nrrd examples", "XCAT_liver","patient_liver_nrrd", "Dominik_XCAT_liver"]:
		patients = os.listdir(root)
		if not dataset["patient_dirs"]: patients = [""]
		
		for patient in patients:
			patient_file_count = 0
			for file in os.listdir(os.path.join(root, patient)):	
				if file.endswith(".nrrd") or file.endswith(".NRRD") or file.endswith(".nii.gz") or file.endswith(".NII.GZ"):
					total_file_count += 1
					patient_file_count += 1

					filePath = os.path.join(root, patient, file)
					file_id = patient_file_count
					if dataset["patient_dirs"]: 
						custom_meta.append((patient, file_id))
					else: 
						custom_meta.append((file_id))
					filePaths.append(filePath)

	elif dataset["name"] in [""]:
		patients = os.listdir(root)
		if not dataset["patient_dirs"]: patients = [""]

		for patient in patients:
			patient_file_count = 0
			# Go through all folders while saving the name in metadata for identification
			for weighting in os.listdir(os.path.join(root, patient)):
				if dataset["weighting"] in weighting:
					for file in os.listdir(os.path.join(root, patient, weighting)):	
						if file.endswith(".nrrd") or file.endswith(".NRRD"):
							total_file_count += 1
							patient_file_count += 1
							#print(patient, weighting, file)
							filePath = os.path.join(root, patient, weighting, file)
							#print(filePath)
							file_id = file[-9:-5]
							custom_meta.append((patient, file_id))
							#print(custom_meta)
							filePaths.append(filePath)

	total_patients = len(patients)
	if verbose: print("\nFound %d files from %d patients" % (total_file_count, total_patients))

	example_image = preprocessing.load_image(Path(filePaths[0]))
	example_image2 = preprocessing.load_image(Path(filePaths[len(filePaths)//2]))
	example_image3 = preprocessing.load_image(Path(filePaths[-1]))
	if verbose: 
		print("\nFirst image:\t(image size: %s  \t| spacing: %s\t)" % (str(example_image.GetSize()), str(example_image.GetSpacing())))
		print("Middle image:\t(image size: %s \t| spacing: %s\t)" % (str(example_image2.GetSize()), str(example_image2.GetSpacing())))
		print("Last image:\t(image size: %s \t| spacing: %s\t)\n" % (str(example_image3.GetSize()), str(example_image3.GetSpacing())))

	return filePaths, custom_meta


def get_dict(root, dataset, verbose = False):
	'''
	Get image-filepaths in necessary dictionary structure in 
	order to be used by the preprocess_dataset() function

	used in call_preprocess_dataset()

    Parameters
    ----------
    root : string
        string filepath to root folder of dataset
    dataset: dict
    	dataset dictionary containing modality, filetype, relative path and weighting
    shuffle : boolean
    	shuffle the images or not
    amount : int
    	amount of patients to read out
    	0 = all
	verbose : boolean
    	Step by step description in terminal

    Returns
    -------
    dict
        Dict containing a key and a dict for each image. 
        The key contains all information needed for identification and the dict 
        is made to be used later with the preprocessing function.
	'''

	filePaths, meta = get_filePaths(root, dataset, verbose)
	if verbose: print("(2/5 Preprocessing) Write paths into dictionary")

	# Fill the dictionary
	images_dict = {}
	for ID, path in enumerate(filePaths):
		
		if dataset["name"] in ["ckm t2 ima", "ckm t1 ima"]:
			patient_id, file_id = meta[ID]
			key = "patient-%s__weigh-%s__img-%s__mod-%s" % (patient_id, dataset["weighting"], file_id, dataset["modality"])

		elif dataset["name"] in ["nrrd examples", "XCAT_liver","patient_liver_nrrd", "Dominik_XCAT_liver"]:
			if dataset["patient_dirs"]:
				patient_id, file_id = meta[ID]
				key = "patient-%s__img-%s" % (patient_id, file_id)
			else:
				file_id = meta[ID]
				key = "img-%s" % (file_id)
		
		images_dict[key] = {"images": [Path(path)]}

	return images_dict
		

def call_preprocess_dataset(preprocessed_root, dataset, verbose = False):
	'''
	Calls the preprocess_dataset() function. 
	Returns the preprocessed dataset in dictionary form as well as saves every preprocessed image in the designated folder as ".nii.gz"

	Parameters
    ----------
    preprocessed_root : string
        string filepath to root folder of folders containing preprocessed data
    dataset: dict
    	dataset dictionary containing modality, filetype, relative path and weighting
	n_patients: int
		amount of patients to process, 0 for all available
	verbose : boolean
    	Step by step description in terminal

    Returns
    -------
    dict
        A new dictionary containing the processed images. All keys in the dict
        for each individual patient are added to the new dict as well. The keys are
        kept the same also.
	'''

	imgs_dict = get_dict(os.path.join(dataset["path"]), dataset, verbose)

	if verbose: print("(3/5 Preprocessing) Preprocess data")

	prepro_params = dict(normalizing_method		 	= normalization.NORMALIZING.QUANTILE,
						 normalization_parameters	= dict(lower_q=0.05, upper_q=0.95),
						 resample				 	= True,
						 target_spacing				= (1, 1, None)) # The spacing for the resampling, by default (1, 1, 3), if any values are None, the original spacing is used

	# Create indiv. folders for preprocessed data
	path_name = "preprocessed_data-%s/" % (dataset["name"])
	Path(os.path.join(preprocessed_root, path_name)).mkdir(parents=True, exist_ok=True)
	prepro_dict = preprocessing.preprocess_dataset(data_set 				= imgs_dict, 
												   num_channels 			= 1, 
												   base_dir 				= Path(dataset["path"]), 
		   										   preprocessed_dir 		= Path(os.path.join(preprocessed_root, path_name)),
		   										   train_dataset 			= (), 
		   										   preprocessing_parameters = prepro_params)
	return prepro_dict


def train_val_test_split(data_dict, val_frac = 0.2, test_frac = 0.2, random_state = 1, verbose = False):
	'''
	Splits the data given into training, validation and test set

	Parameters
    ----------
    data_dict : dict
        same style as output from preprocess_dataset()
    val_frac: string
    	fraction of all data that will be separated as validation data
	test_frac: string
    	fraction of all data that will be separated as testing data
	random_state: int
		State to reproduce random data
    verbose : boolean
    	Step by step description in terminal

    Returns
    -------
    Dict
        Dictionary of all data used for training
    Dict
    	Dictionary of all data used for validation
	Dict
    	Dictionary of all data used for testing
	'''
	if verbose: print("(4/5 Preprocessing) Split train/val/test data")
	data_list = list(data_dict.items())

	# X_train, X_val = train_test_split(data_list, test_size=val_frac, random_state=random_state)

	X_train, X_test = train_test_split(data_list, test_size=test_frac, random_state=random_state)
	X_train, X_val = train_test_split(X_train, test_size=val_frac/(1-test_frac), random_state=random_state)

	X_train_dict = dict(X_train)
	X_val_dict = dict(X_val)
	X_test_dict = dict(X_test)

	return X_train_dict, X_val_dict, X_test_dict


def get_tf_datasets(preprocess_root, train_data, val_data, test_data, input_shape, batch_size, samples_per_volume, n_epochs, verbose = False):
	"""
    Preprocesses the images from the preprocessed dictionary and returns a tensorflow dataset 
    split into training and validation dataset. 

    Parameters
    ----------
    preprocess_root: string
    	root dictionary of all data, train_data/val_data are rel. to that dir
    train_data: dict
    	same style as output from preprocess_dataset()
	val_data: dict
		same style as output from preprocess_dataset()
	test_data: dict
		same style as output from preprocess_dataset()
    batch_size: int
        batch size of the returned tensorflow dataset
    samples_per_volume: int
        how many images shall be sampled
    n_epochs: int
        the number of epochs
	verbose : boolean
    	Step by step description in terminal

    Returns
    -------
    tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int]
        a tuple consisting out of
        - a tensorflow dataset with the preprocessed training data
        - a tensorflow dataset with the preprocessed validation data
        - a tensorflow dataset with the preprocessed testing data
        - the number of steps per epoch in training
    """
	if verbose: print("(5/5 Preprocessing) Load data into tf.data.Datasets\n")

	ckm_cfg.train_input_shape = cfg.image_shape
	ckm_cfg.num_channels = cfg.image_shape[2]

	# Crucial, otherwise get_filenames in segbasisloader.py will see cfg.data_base_dir as None and raise an error
	ckm_cfg.data_base_dir = preprocess_root

    # Define loaders
	training_loader = segbasisloader.SegBasisLoader(file_dict 			= train_data, # Jonathan hat hier alle nicht nur die train_data 
													mode 				= segbasisloader.SegBasisLoader.MODES.TRAIN, 
													frac_obj 			= 0, 
													samples_per_volume 	= samples_per_volume,
													tasks 				= "other")

	valid_loader = segbasisloader.SegBasisLoader(file_dict 				= val_data,
												 mode 					= segbasisloader.SegBasisLoader.MODES.VALIDATE,
	    										 frac_obj 				= 0,
	    										 samples_per_volume 	= samples_per_volume,
	    										 tasks 					= "other")

	test_loader = segbasisloader.SegBasisLoader(file_dict 				= test_data,
												 mode 					= None,
	    										 frac_obj 				= 0,
	    										 samples_per_volume 	= samples_per_volume,
	    										 tasks 					= "other")

	# training_loader/val_loader require a list of key-strings and not the given dict of dicts
	train_key_list 	= [str(key) for key in list(train_data.keys())]
	val_key_list 	= [str(key) for key in list(val_data.keys())]
	test_key_list 	= [str(key) for key in list(test_data.keys())]

	# Load datasets
	training_dataset 	= training_loader(train_key_list, batch_size, n_epochs)
	valid_dataset 		= valid_loader(val_key_list, batch_size, n_epochs)
	test_dataset 		= test_loader(test_key_list, batch_size, n_epochs)

	steps_per_epoch = len(train_data) * samples_per_volume // batch_size

	return [training_dataset, valid_dataset, test_dataset, steps_per_epoch]




def load_datasets(dataset_name):
	dataset_info = cfg.all_datasets[dataset_name]
	
	if cfg.data_loader_verbose: print("\n___________(Loading %s dataset | mod: %s | type: %s | weighting: %s)___________"
		%(dataset_info["name"], dataset_info["modality"].upper(), 
			dataset_info["weighting"]))
	else:
		if cfg.verbose: print("\n================= Loading dataset: %s =================\n" % (dataset_info["name"]))
	
	if cfg.image_count>0:
		preprocessed_dict = dict(take(cfg.image_count, call_preprocess_dataset(cfg.local_data_root, dataset_info, cfg.data_loader_verbose).items()))
		if cfg.verbose: print("\nLoading %s images!"%(cfg.image_count))
	else:
		if cfg.verbose: print("\nLoading all images!")
		preprocessed_dict = call_preprocess_dataset(cfg.local_data_root, dataset_info, cfg.data_loader_verbose)

	train_data, val_data, test_data = train_val_test_split(preprocessed_dict, val_frac = 0.2, test_frac = 0.2)
	[train_ds, val_ds, test_ds, cfg.steps_per_epoch] = get_tf_datasets(cfg.local_data_root, train_data, 
		val_data, test_data, cfg.image_shape, cfg.batch_size, cfg.samples_per_volume, cfg.n_epochs, cfg.data_loader_verbose)

	if cfg.verbose: print("\n%d training images loaded!" % (len(train_data)))
	if cfg.verbose: print("%d val images loaded!" % (len(val_data)))
	if cfg.verbose: print("%d test images loaded!\n" % (len(test_data)))

	return train_ds, val_ds, test_ds
