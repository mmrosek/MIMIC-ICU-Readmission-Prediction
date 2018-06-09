import os

FULL_LABEL_SIZE = 8922
FULL_LABEL_SIZE_II = 5031

PAD_CHAR = "**PAD**"
EMBEDDING_SIZE = 10
MAX_LENGTH = 250

#where you want to save any models you may train
abs_path = os.path.abspath(__file__)
file_dir = os.path.dirname(abs_path)
parent_dir = os.path.dirname(file_dir)
parent_dir2 = os.path.dirname(parent_dir)
parent_dir3 = os.path.dirname(parent_dir2) + "/Models"

print("\nMODEL DIR: " + str(parent_dir3))

MODEL_DIR = parent_dir3

DATA_DIR = '../mimicdata/'
MIMIC_3_DIR = '../mimicdata/mimic3'
MIMIC_2_DIR = '/mimicdata/mimic2'
