import os
from os import listdir, makedirs
from os.path import isfile, join
from termcolor import colored
import shutil


# os.system("qsub -I -V -N kukk -l nodes=1:gpus=8:V100")


_FOLDER_ROOT_ = "/Users/i337936/Documents/shany_net/shany_net/"
_FOLDER_PROJECT_ = _FOLDER_ROOT_ + "dataset_final_project/"
_FOLDER_DATASET_ = _FOLDER_PROJECT_ + "dataset/"
_FOLDER_TRAIN_ = _FOLDER_PROJECT_ + "train/"
_FOLDER_INFERENCE_ = _FOLDER_PROJECT_ + "inference/"
_FOLDER_VALIDATON_ = _FOLDER_PROJECT_ + "validation/"
_FOLDER_MODEL_ = _FOLDER_PROJECT_ + "model/"
_FOLDER_ZIP_ = _FOLDER_ROOT_ + "zip/"
_ZIP_OUTPUT_ = _FOLDER_ZIP_ + "dataset_final_project"  #.zip will be appended automatically

def zipdir(path, ziph):
	# ziph is zipfile handle
	for root, dirs, files in os.walk(path):
		for file in files:
			ziph.write(os.path.join(root, file))

def remove_folder(folder = None):
	if folder is None:
		return

	shutil.rmtree(folder)

def create_folder(folder = None):
	if folder is None:
		return

	if not os.path.exists(folder):
		os.makedirs(folder)

def gen_data(folder_path=None, output = None, percent = 0.8):
	folders = [f for f in listdir(folder_path) if not isfile(join(folder_path, f))]

	for folder in folders:
		print(colored('Preparing '+folder_path+'', 'white') + " -> " + colored(output, 'green'))
		files = [f for f in listdir(folder_path + folder) if isfile(join(folder_path + folder, f))]

		total_files = len(files)
		total = round(total_files * percent)
		files = files[:total]

		for file in files:
			print(colored('Moving', 'blue') + " -> " + colored(file, 'red'))
			create_folder(output + folder + "/")
			src = folder_path + folder + "/" + file
			dest = output + folder + "/" + file
			shutil.move(src, dest)

def generate():
	create_folder(_FOLDER_TRAIN_)
	create_folder(_FOLDER_INFERENCE_)
	create_folder(_FOLDER_VALIDATON_)
	create_folder(_FOLDER_MODEL_)
	gen_data(_FOLDER_DATASET_, _FOLDER_TRAIN_, 0.8)  # generate train data from dataset source
	gen_data(_FOLDER_DATASET_, _FOLDER_INFERENCE_, 1)  # generate inference data from dataset source
	gen_data(_FOLDER_TRAIN_, _FOLDER_VALIDATON_, 0.2)  # generate validation data from train source

def clear_data():
	remove_folder(_FOLDER_DATASET_)

def create_zip():
	create_folder(_FOLDER_ZIP_)
	print("Creating project zip file ...", end=" ", flush=True)
	shutil.make_archive(_ZIP_OUTPUT_, 'zip', _FOLDER_PROJECT_)
	shutil.move(_FOLDER_ZIP_, _FOLDER_PROJECT_)
	print("Done.")

if __name__ == '__main__':
	# generate()
	# clear_data()
	# create_zip()
