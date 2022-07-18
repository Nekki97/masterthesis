import keras


########################################## Dataset Parameters ##########################################
root = "C:/Users/nekta/Documents/University/Masterarbeit/ActualArbeit/"

code_root 		= root+"Code/MyCode"
local_data_root = root+"Data/"
output_root		= root+"Output/"


test_frac = 0.2
val_frac = 0.2

all_datasets = {"XCAT_liver": 				{"ID"			: 1,
											"name"			: "XCAT_liver",
											"keyword"		: "",
											"patient_dirs"	: False,
											"path"			: root+"J-Datasets/genT1_XCAT_liver"},

				"patient_liver_nrrd": 		{"ID"			: 2,
											"name"			: "patient_liver_nrrd",
											"keyword"		: "t1_vibe_opp-in_tra",
											"patient_dirs"	: False,
											"path"			: local_data_root+"/patient_liver_conv_nrrd"},

				"Dominik_XCAT_liver":		{"ID"			: 3,
											"name"			: "Dominik_XCAT_liver",
											"keyword"		: "",
											"patient_dirs"	: False,
											"path"			: root+"J-Datasets/Dominik_genT1_XCAT_liver"},

				"postVIBE_XCAT":			{"ID"			: 4,
											"name"			: "postVIBE_XCAT",
											"keyword"		: "",
											"patient_dirs"	: False,
											"path"			: root+"J-Datasets/postVIBE_XCAT"}}


####################################### Data Augmentation Parameters ###################################
augmentation_cfg = {"gaussian_noise_only":	{"ID"					: 1,
											"add_noise" 			: True,
										  	 "noise_type"			: "gaussian",
										  	 "standard_deviation"	: 0.01,
										  	 "mean_poisson"			: 0.5},
			  	    "poisson_noise_only":	{"ID"					: 2,
			  	    						 "add_noise" 			: True,
										  	 "noise_type"			: "poisson",
										  	 "standard_deviation"	: 0.01,
										  	 "mean_poisson"			: 0.5}}


# Only work if input size is smaller than original
randomly_crop 	= True
augment_crop 	= True # Augment by randomly cropping same image multiple times
n_aug_crops 	= 4 # How often 

####################################### Keras CycleGAN Parameters ###################################
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


# generator_loss_options = {"id"	: {"ID"		: 1, "identity"	: True,  "gradient"	: False, "percept"	: False},
# 						 "gr"	: {"ID"		: 2, "identity"	: False, "gradient"	: True,  "percept"	: False},
# 						 "comb"	: {"ID"		: 3, "identity"	: True,  "gradient"	: True,  "percept"	: False},
# 						 "perc"	: {"ID"		: 4, "identity"	: False, "gradient"	: False, "percept"	: True}}

models_cfg = {"VanillaCycleGAN" 	: {"ID" : 1},	# VanillaCycleGAN: 	basic discriminator, 	resnet generator
			  "AdvancedCycleGAN" 	: {"ID"	: 2},	# AdvancedCycleGAN: patchgan discriminator, unet generator
			  "DominikCycleGAN"		: {"ID"	: 3}}	# DominikCycleGAN: 	patchgan discriminator, resnet generator

loss_weight_cfg = {"1":{"ID":1,"cycle":10,"identity":0.5,"gradient":0,  "perception":0}, # Vanilla
				   "2":{"ID":2,"cycle":10,"identity":0.4,"gradient":0.4,"perception":0}, # Dominiks Code
			   	   "3":{"ID":3,"cycle":10,"identity":0,  "gradient":0,  "perception":0},
			   	   "4":{"ID":5,"cycle":10,"identity":5,  "gradient":0,  "perception":0},
			   	   "5":{"ID":6,"cycle":10,"identity":10,  "gradient":0,  "perception":0}}

########################################## Training Parameters #########################################
batch_size 			= 2
# samples_per_volume = 4
n_epochs 			= 100
image_count 		= 200 # 0 for all available

# !!! when changing make sure preprocess is True !!! 
image_shape 		= [128, 128, 1] # data input shape

save_period 		= 5


verbose 			= True

preprocess			= True	# if False, takes saved preprocessed data
augment 			= False # Does nothing rn

visualize_data 		= True
train 				= True
test 				= True

########################################## Loop Configurations #########################################

# Gen F generates first in pair, Gen G generates second in pair
dataset_pairs 			= [["postVIBE_XCAT", 		"patient_liver_nrrd"]]
						   #["Dominik_XCAT_liver", 	"patient_liver_nrrd"]]
# generator_losses 		= ["comb"]
models 					= ["DominikCycleGAN", "AdvancedCycleGAN"]
loss_weights			= ["2", "1"]

# augmentation_strategies = ["gaussian_noise_only"]

# TODO: make list of trial configuration --> run one by one for better customizability

# All data will be saved in folder named:
trial_name 				= "test_single_channel"


########################################################################################################

# ignore

trial_settings = {"modelID"				: None,
				  "modelname"			: None,
				  "datasetGID"			: None,
				  "datasetFID"			: None,
				  "steps_per_epoch"		: None,
				  "training_img_shapes"	: None, 
				  "loss_weights"		: "1"}

				  #"g_lossID"			: None,
				  #"g_loss"				: None,



# git add .
# git commit -m "message"
# git push -u origin single_channel

# git switch -c newbranchname