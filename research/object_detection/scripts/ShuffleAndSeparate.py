import os
import random
import glob 
import shutil
import sys
import pandas as pd
from tqdm import tqdm




def WriteListToFile(info_list:list, filename):
    ## Write the List to a .csv file
    ## Note: Since only one list, dataframe is not used 

    with open(filename, "w") as outfile:
        for item in info_list:
            outfile.write(item)
            outfile.write("\n")

def ShuffleAndSeparate():
    ROOT_DIR = os.getcwd()

    input_img_arg = sys.argv[1]
    input_xml_arg = sys.argv[2]
    input_test_ratio = sys.argv[3]
    if not input_img_arg:
        IMG_DIR = os.path.join(ROOT_DIR, "img")
    else:
        IMG_DIR = os.path.join(ROOT_DIR, input_img_arg)
    
    if not input_xml_arg:
        XML_DIR = os.path.join(ROOT_DIR, "xml")
    else:
        XML_DIR = os.path.join(ROOT_DIR, input_xml_arg)
    
    if not input_test_ratio:
        input_test_ratio = 0.1

    TEST_DATASET_DIR = os.path.join(ROOT_DIR,"test")
    TRAIN_DATASET_DIR = os.path.join(ROOT_DIR, "train")

    ## Check if the Corresponding Directory exists 
    if not os.path.exists(IMG_DIR):
        raise ValueError("[Error] Directory named {} is not found. Please Check".format(os.path.basename(IMG_DIR)))

    if not os.path.exists(XML_DIR):
        raise ValueError("[Error] Directory named {} is not found. Please Check".format(os.path.basename(XML_DIR)))

    if not os.path.exists(TEST_DATASET_DIR):
        os.makedirs(TEST_DATASET_DIR)

    if not os.path.exists(TRAIN_DATASET_DIR):
        os.makedirs(TRAIN_DATASET_DIR)

    IMG_FILE_LIST = [f for f in glob.glob(IMG_DIR + "/*.jpg", recursive=True)]
    XML_FILE_LIST = [f for f in glob.glob(XML_DIR + "/*.xml", recursive=True)]

    ## Check if the img dir is empty
    if not (IMG_FILE_LIST):
        raise ValueError("[Error] Directory named {} has no .jpg image. Please Check".format(os.path.basename(IMG_DIR)))
    
    ## Check if the xml dir is empty
    if not (XML_FILE_LIST):
        raise ValueError("[Error] Directory named {} has no .xml file. Please Check".format(os.path.basename(XML_DIR)))
    
    ## Check if the img dir is empty
    if not len(IMG_FILE_LIST) == len(XML_FILE_LIST):
        raise ValueError("[Error] The number of .jpg image is not match with the number of .xml file. Please Check")

    print("[Processing] {} files are found in directory named {}"\
        .format(len(IMG_FILE_LIST), os.path.basename(IMG_DIR)))
    print("[Processing] {} files are found in directory named {}"\
        .format(len(XML_FILE_LIST), os.path.basename(XML_DIR)))

    ## Shuffle the XML List
    random.shuffle(XML_FILE_LIST)

   

    test_file_num = round(float(input_test_ratio) * len(XML_FILE_LIST))

    train_set = []
    test_set = []

    ## Split the dataset into two part: train_set & test_set
    for i in range(test_file_num):
        elem = XML_FILE_LIST.pop()
        test_set.append(elem)

    train_set = [f for f in XML_FILE_LIST]

    WriteListToFile(train_set, "train.csv")
    WriteListToFile(test_set, "test.csv")


    print("[Processing] {} files in test set".format(len(test_set)))
    print("[Processing] {} files in train_set".format(len(train_set)))

    print("[Processing] Moving of test dataset")
    pbar = tqdm(test_set)
    for iItems,  f in enumerate(pbar):
        shutil.copy(f,TEST_DATASET_DIR)
        shutil.copy(os.path.join(IMG_DIR, os.path.basename(f).replace(".xml",".jpg")), TEST_DATASET_DIR)
        pbar.set_description("Items: %.2f" %iItems)


    print("[Processing] Moving of train dataset")
    pbar = tqdm(train_set)
    for iItems,  f in enumerate(pbar):
        shutil.copy(f,TRAIN_DATASET_DIR)
        shutil.copy(os.path.join(IMG_DIR, os.path.basename(f).replace(".xml",".jpg")), TRAIN_DATASET_DIR)
        pbar.set_description("Items: %.2f" %iItems)

    print("[Processing] Finish Process")


def main():
    ShuffleAndSeparate()

if __name__ == '__main__':
    main()