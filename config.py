import keras


########################################## Dataset Parameters ##########################################
root = "C:/Users/nekta/Documents/University/Masterarbeit/ActualArbeit/"

code_root 		= root+"Code/MyCode"
local_data_root = root+"Data/"
output_root		= root+"Output/"


test_frac = 0.2
val_frac = 0.2
normalization_ID = "1"

normalization_cfg = {"1" : [0,1],
					 "2" : [-1,1]}

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
			   	   "5":{"ID":6,"cycle":10,"identity":10, "gradient":0,  "perception":0}}

########################################## Training Parameters #########################################
batch_size 			= 1

# samples_per_volume = 4

n_epochs 			= 100

# How many different files to read out data from (will be cut down to an equal amount of each file for max. generalizability)
n_files 			= 0 	# 0 for all available (56 XCAT models)

# Total image count in training dataset (steps per epoch = imagecount / batchsize) 
# --> keras automatically stops when one dataset runs out of data => image_count = 0 means train until smaller dataset runs out
image_count 		= 200 	# 0 for all available

# !!! when changing make sure preprocess is True !!! 
image_shape 		= [256, 256, 1] # data input shape


save_period 		= 5 	# Save model and generate images every x epochs


verbose 			= True

# Takes very long if all files are loaded, load files once and then use the generated preprocessed dataset by setting this to False
preprocess			= False	# if False, takes saved preprocessed data

augment 			= False # Does nothing rn

visualize_data 		= True # Visualize sample data before training, needs user input to continue (otherwise everything runs alone)
train 				= True
test 				= True

########################################## Loop Configurations #########################################

# Gen F generates first in pair, Gen G generates second in pair
# Put the dataset with labels in first place
# TODO: make so it doesnt matter in which position the labelled one is 
dataset_pairs 			= [["postVIBE_XCAT", "patient_liver_nrrd"]]
						  #["Dominik_XCAT_liver", 	"patient_liver_nrrd"]]
# generator_losses 		= ["comb"]
models 					= ["DominikCycleGAN", "AdvancedCycleGAN"]
loss_weights			= ["2"]

xcat_organ_mask			= "liver"

# augmentation_strategies = ["gaussian_noise_only"]

# TODO: make list of trial configuration --> run one by one for better customizability

# All data will be saved in folder named:
trial_name 				= "more_tests"

# TODO: write "-2" in folder name if folder already exists


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




# Values in Dominiks MR-XCAT
organ_mask_vals = 	{"liver" 		: {"Model71": 5.147,"Model76": 5.115,"Model77": 4.809,"Model80": 5.275,
									   "Model86": 4.872,"Model89": 4.991,"Model92": 5.069,"Model93": 5.339,
									   "Model96": 5.113,"Model98": 5.082,"Model99": 5.380,"Model106":5.151,
									   "Model108":5.025,"Model117":5.146,"Model118":5.257,"Model128":5.314,
									   "Model139":5.011,"Model140":5.025,"Model141":5.260,"Model142":5.138,
									   "Model143":5.165,"Model144":4.851,"Model145":5.029,"Model146":4.917,
									   "Model147":4.856,"Model148":5.066,"Model149":5.081,"Model150":4.716,
									   "Model151":5.210,"Model152":4.999,"Model153":5.269,"Model154":5.220,
									   "Model155":5.267,"Model157":5.068,"Model159":4.942,"Model162":4.999,
									   "Model163":4.997,"Model164":5.059,"Model166":4.852,"Model167":5.242,
									   "Model168":5.121,"Model169":5.317,"Model170":5.044,"Model171":5.249,
									   "Model173":5.237,"Model175":5.149,"Model176":5.130,"Model178":5.193,
									   "Model180":4.687,"Model182":5.214,"Model184":4.889,"Model196":5.283,
									   "Model200":5.084,"Model201":5.122,"Model401":5.037,"Model447":5.029},
					 "right_kidney"	: {},
					 "left_kidney" 	: {}}


# git add .
# git commit -m "message"
# git push -u origin single_channel

# git switch -c newbranchname