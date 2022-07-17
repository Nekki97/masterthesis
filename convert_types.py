import SimpleITK as sitk
import sys
import os
from pathlib import Path


def convert_ima_to_nrrd_series(root, out_path, mod):
	Path(os.path.join(out_path)).mkdir(parents=True, exist_ok=True)

	for patientID in os.listdir(os.path.join(root)):
		for modality in os.listdir(os.path.join(root,patientID)):
			if modality == mod:
				for file in os.listdir(os.path.join(root,patientID,mod)):
					in_path = os.path.join(root, patientID, modality, file)
					image = sitk.ReadImage(in_path)

					fileID = file[12+int(int(patientID)>9):16+int(int(patientID)>9)]
					path = os.path.join(out_path, patientID)
					Path(os.path.join(path)).mkdir(parents=True, exist_ok=True)
					sitk.WriteImage(image, os.path.join(path,"%s.nrrd"%(fileID)))


def convert_liver_dcm_to_nrrd_series(path_in, path_out, target_weighting):
	Path(os.path.join(path_out)).mkdir(parents=True, exist_ok=True)
	for patientID in os.listdir(path_in):
		for folder in os.listdir(os.path.join(path_in, patientID)):
			if os.path.isdir(os.path.join(path_in, patientID, folder)):
				for weighting in os.listdir(os.path.join(path_in, patientID, folder)):
					if target_weighting in weighting:
						path = os.path.join(path_in, patientID, folder, weighting)

						files = [i for i in os.listdir(path)]
						series_ID = [files[0][3:7], files[-1][3:7]]
						
						series_list = [[], []]

						for file in files:
							if file[3:7] == series_ID[0]:
								series_list[0].append(os.path.join(path,file))
							elif file[3:7] == series_ID[1]:
								series_list[1].append(os.path.join(path,file))

						for i, series in enumerate(series_list):
							series_reader = sitk.ImageSeriesReader()
							series_reader.SetFileNames(series)
							series_reader.LoadPrivateTagsOn()
							image = series_reader.Execute()

							sitk.WriteImage(image, os.path.join(path_out,"%s-%s.nrrd"%(patientID[-8:], series_ID[i])))

						#path = os.path.join(path_in, patientID, folder, weighting, file)
						#image = sitk.ReadImage(path)
						#fileID = file[-13:-9]
						#out_path = os.path.join(path_out, patientID[-8:], weighting)
						#Path(os.path.join(out_path)).mkdir(parents=True, exist_ok=True)
						#sitk.WriteImage(image, os.path.join(out_path,"%s_%s.nrrd"%(patientID[-8:],fileID)))					

'''
path_in = "J:/AG_IMAGIN/01_Projects/21_RegistrierungsGT/04_Data/M2Olie_new/M2olie_Patientdata_dcm/M2OLIE3"
path_out = "C:/Users/nw17/Documents/Masterthesis/Data/patient_liver_conv_nrrd"
weighting = "t1_vibe_in-opp_tra"
convert_liver_dcm_to_nrrd_series(path_in, path_out, weighting)
'''


'''
data_root = "J:/AG_IMAGIN/05_Data sets/GRASP/t1 t2 Data/"
target_modality = "t1_vibe_dixon_cor_W"
path_out = "C:/Users/nw17/Documents/Masterthesis/Data/patient_kidney_conv_nrrd"

convert_to_nrrd_series(data_root, path_out, target_modality)
'''