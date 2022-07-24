import keras


all_machines = {"1": "mein ckm pc",
				"2": "ckm server",
				"3": "pc daheim",
				"4": "laptop"}

########################################## Dataset Parameters ##########################################

# Define which machine is used to automatically set the paths

###############
machine = "1" #
###############

if all_machines[machine] == "mein ckm pc":
	root = "C:/Users/nw17/Documents/Masterthesis/"
elif all_machines[machine] == "pc daheim":
	root = "C:/Users/nekta/Documents/University/Masterarbeit/ActualArbeit/"

test_frac = 0.2
val_frac = 0.2
normalization_ID = "1"

####################################### Data Augmentation Parameters ###################################
# Only work if input size is smaller than original
randomly_crop 	= True
augment_crop 	= True # Augment by randomly cropping same image multiple times
n_aug_crops 	= 4 # How often 

####################################### Keras CycleGAN Parameters ###################################
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


########################################## Training Parameters #########################################
batch_size 			= 1
n_epochs 			= 5

# How many different files to read out data from (will be cut down to an equal amount of each file for max. generalizability)
n_files 			= 3 	# 0 for all available (56 XCAT models)

# Total image count in training dataset (steps per epoch = imagecount / batchsize) 
# --> keras automatically stops when one dataset runs out of data => image_count = 0 means train until smaller dataset runs out
image_count 		= 5 	# 0 for all available

# !!! when changing make sure preprocess is True !!! 
image_shape 		= [64, 64, 1] # data input shape
save_period 		= 1 	# Save model and generate images every x epochs

# Takes very long if all files are loaded, load files once and then use the generated preprocessed dataset by setting this to False
preprocess			= True	# if False, takes saved preprocessed data
verbose 			= True
visualize_data 		= False # Visualize sample data before training, needs user input to continue (otherwise everything runs alone)
train 				= True
test 				= True

########################################## Loop Configurations #########################################
xcat_dataset_name 		= "postVIBE_XCAT"
xcat_organ_mask			= "liver"
patient_dataset_name 	= "patient_liver_nrrd"

# Loop over settings
loop_settings			= {1:{"model_name"		:"DominikCycleGAN", 
							  "loss_weights_ID"	:"4"},

						   2:{"model_name"		:"DominikCycleGAN", 
						   	  "loss_weights_ID"	:"5"}}

# All data will be saved in folder named:
trial_name 				= "test"

########################################################################################################








# ignore
trial_settings = {"modelID"				: None,
				  "modelname"			: None,
				  "datasetGID"			: None,
				  "datasetFID"			: None,
				  "steps_per_epoch"		: None,
				  "training_img_shapes"	: None, 
				  "loss_weights"		: "1"}

models_cfg = {"VanillaCycleGAN" 	: {"ID" : 1},	# VanillaCycleGAN: 	basic discriminator, 	resnet generator
			  "AdvancedCycleGAN" 	: {"ID"	: 2},	# AdvancedCycleGAN: patchgan discriminator, unet generator
			  "DominikCycleGAN"		: {"ID"	: 3}}	# DominikCycleGAN: 	patchgan discriminator, resnet generator

loss_weight_cfg = {"1":{"ID":1,"cycle":10,"identity":0.5,"gradient":0,  "perception":0}, # Vanilla
				   "2":{"ID":2,"cycle":10,"identity":0.4,"gradient":0.4,"perception":0}, # Dominiks Code
			   	   "3":{"ID":3,"cycle":10,"identity":0,  "gradient":0,  "perception":0},
			   	   "4":{"ID":5,"cycle":10,"identity":5,  "gradient":0,  "perception":0},
			   	   "5":{"ID":6,"cycle":10,"identity":10, "gradient":0,  "perception":0}}


normalization_cfg = {"1" : [0,1],
					 "2" : [-1,1]}


code_root 		= root+"Code/MyCode"
local_data_root = root+"Data/"
output_root		= root+"Output/"

all_datasets = {"XCAT_liver": 				{"ID"			: 1,
											"name"			: "XCAT_liver",
											"keyword"		: "",
											"patient_dirs"	: False,
											"path"			: local_data_root+"/genT1_XCAT_liver"},

				"patient_liver_nrrd": 		{"ID"			: 2,
											"name"			: "patient_liver_nrrd",
											"keyword"		: "t1_vibe_opp-in_tra",
											"patient_dirs"	: False,
											"path"			: local_data_root+"/patient_liver_conv_nrrd"},

				"Dominik_XCAT_liver":		{"ID"			: 3,
											"name"			: "Dominik_XCAT_liver",
											"keyword"		: "",
											"patient_dirs"	: False,
											"path"			: local_data_root+"/Dominik_genT1_XCAT_liver"},

				"postVIBE_XCAT":			{"ID"			: 4,
											"name"			: "postVIBE_XCAT",
											"keyword"		: "",
											"patient_dirs"	: False,
											"path"			: local_data_root+"/postVIBE_XCAT"}}

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