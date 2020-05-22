import os
import random
import glob 
import shutil
import sys
import pandas as pd
from tqdm import tqdm
import cv2


#############################################
#   Global Varialble
#############################################
# Default
# label_dictionary ={
#     'PIN': 0,
#     'BODY': 1
# }
# label_dictionary ={
#     'PIN': 2,
# }
label_dictionary ={
    'PIN': 0,
    'BODY': 1,
    'PIN_NL': 2,
    'PIN_FLAT': 3,
    'PIN_GULL': 4,
    'PIN_JLEAD':5
    
}
reverse_dictionary = dict([(value, key) for (key, value) in label_dictionary.items()])


#############################################
#   Function Definition
#############################################
def WriteListToFile(info_list:list, filename, mode="w"):
    ## Write the List to a .csv file
    ## Note: Since only one list, dataframe is not used 

    with open(filename, mode) as outfile:
        for item in info_list:
            outfile.write(item) 
            outfile.write("\n")


def read_as_list(fname):

    # Open .txt file containing dimensions
    # filename,width,height,class,xmin,ymin,xmax,ymax
    with open(fname) as f:
        content = f.readlines()

    # Jpg
    jpg_name = fname.replace('.txt', '.jpg')
    img = cv2.imread (jpg_name)
    h, w, _ = img.shape
    base_name = os.path.basename(fname).replace('.txt', '.jpg')
    all_lines = []

    # Content
    for r in content:
        try:
            xmin, ymin, xmax, ymax, class_int = list(map(lambda x: int(x), r.strip().split(',')))
            class_lbl = reverse_dictionary[class_int]
        except :
            xmin, ymin, xmax, ymax = list(map(lambda x: int(x), r.strip().split(',')[:-1] ))
            class_lbl = r.strip().split(',')[-1].upper()
            class_int = label_dictionary [class_lbl]
        
        
        line = "{},{},{},{},{},{},{},{}".format(base_name, w, h, class_lbl, xmin, ymin, xmax, ymax)
        all_lines.append(line)

    # Formatting
    return all_lines


def ShuffleAndSeparateForEightChannel(root_dir):


    try:
        input_img_arg = sys.argv[1]
        IMG_DIR = os.path.join(ROOT_DIR, input_img_arg)
    except:
        IMG_DIR = os.path.join(ROOT_DIR, "img")
        print("Using default IMG_DIR = {}".format(IMG_DIR))
    
    try:
        input_txt_arg = sys.argv[2]
        txt_DIR = os.path.join(ROOT_DIR, input_txt_arg)
    except:
        txt_DIR = os.path.join(ROOT_DIR, "txt")
        print("Using default txt_DIR = {}".format(txt_DIR))
    
    try:
        input_test_ratio = sys.argv[3]
    except:
        input_test_ratio = 0.1
        print("Using default input_test_ratio = {}".format(input_test_ratio))

    IMG_DIR=os.path.normpath(IMG_DIR)
    txt_DIR=os.path.normpath(txt_DIR)
    PGM_DIR = IMG_DIR.replace('img', 'pgm')
    WAR_DIR = IMG_DIR.replace('img', 'war')
    TEST_DATASET_DIR = os.path.join(ROOT_DIR,"test")
    TRAIN_DATASET_DIR = os.path.join(ROOT_DIR, "train")

    ## Check if the Corresponding Directory exists 
    if not os.path.exists(IMG_DIR):
        raise ValueError("[Error] Directory named {} is not found. Please Check".format(os.path.basename(IMG_DIR)))

    if not os.path.exists(txt_DIR):
        raise ValueError("[Error] Directory named {} is not found. Please Check".format(os.path.basename(txt_DIR)))

    if not os.path.exists(TEST_DATASET_DIR):
        os.makedirs(TEST_DATASET_DIR)

    if not os.path.exists(TRAIN_DATASET_DIR):
        os.makedirs(TRAIN_DATASET_DIR)

    # Get file list
    txt_FILE_LIST = [f for f in glob.glob(txt_DIR + "/*.txt", recursive=True)]

    ## Check if the txt dir is empty
    if not (txt_FILE_LIST):
        raise ValueError("[Error] Directory named {} has no .txt file. Please Check".format(os.path.basename(txt_DIR)))
    print("[Processing] {} files are found in directory named {}"\
        .format(len(txt_FILE_LIST), os.path.basename(txt_DIR)))

    ## Shuffle the txt List
    random.shuffle(txt_FILE_LIST)
    test_file_num = round(float(input_test_ratio) * len(txt_FILE_LIST))

    train_set = []
    test_set = []

    ## Split the dataset into two part: train_set & test_set
    for i in range(test_file_num):
        elem = txt_FILE_LIST.pop()
        test_set.append(elem)

    train_set = [f for f in txt_FILE_LIST]


    # Overwrite file
    with open (os.path.join(ROOT_DIR, "train_labels.csv"), "w") as fptr:
        fptr.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
    with open (os.path.join(ROOT_DIR, "test_labels.csv"), "w") as fptr:
        fptr.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
    print("[Processing] {} files in test set".format(len(test_set)))
    print("[Processing] {} files in train_set".format(len(train_set)))

    print("[Processing] Moving of test dataset")
    pbar = tqdm(test_set)
    
    for iItems,  f in enumerate(pbar):
        file_basename = os.path.basename(f).replace('.txt','')
        
        for iImg in range(8):

            ext = "_" + str(iImg)
            try:
                shutil.copy(os.path.join(IMG_DIR, file_basename + ext + ".jpg"),                        TEST_DATASET_DIR)
                shutil.copyfile(f,                                                                      os.path.join(TEST_DATASET_DIR, file_basename + ext +'.txt'))
                shutil.copyfile(os.path.join(PGM_DIR, os.path.basename(f).replace(".txt", "_0.pgm")),   os.path.join(TEST_DATASET_DIR, file_basename + ext +'.pgm') )
                shutil.copy(os.path.join(WAR_DIR, os.path.basename(f).replace(".txt", "_0.war")),       os.path.join(TEST_DATASET_DIR, file_basename + ext +'.war') )
            except:
                print("Error copying files:\n {}".format(f))
                sys.exit()

            # Create csv
            boxes = read_as_list(os.path.join(TEST_DATASET_DIR, file_basename + ext +'.txt'))
            WriteListToFile(boxes, os.path.join(ROOT_DIR, "test_labels.csv"), "a")
        
        pbar.set_description("Items: %d" %iItems)


    print("[Processing] Moving of train dataset")
    pbar = tqdm(train_set)
    for iItems,  f in enumerate(pbar):
        file_basename = os.path.basename(f).replace('.txt','')
        
        for iImg in range(8):
            ext = "_" + str(iImg)
            try:
                shutil.copy(os.path.join(IMG_DIR, file_basename + ext + ".jpg"),                        TRAIN_DATASET_DIR)
                shutil.copyfile(f,                                                                      os.path.join(TRAIN_DATASET_DIR, file_basename + ext +'.txt'))
                shutil.copyfile(os.path.join(PGM_DIR, os.path.basename(f).replace(".txt", "_0.pgm")),   os.path.join(TRAIN_DATASET_DIR, file_basename + ext +'.pgm') )
                shutil.copy(os.path.join(WAR_DIR, os.path.basename(f).replace(".txt", "_0.war")),       os.path.join(TRAIN_DATASET_DIR, file_basename + ext +'.war') )
            except:
                print("Error copying files:\n {}".format(f))
                sys.exit()

            # Create csv
            boxes = read_as_list(os.path.join(TRAIN_DATASET_DIR, file_basename + ext +'.txt'))
            WriteListToFile(boxes, os.path.join(ROOT_DIR, "train_labels.csv"), "a")
        pbar.set_description("Items: %d" %iItems)

    print("[Processing] Finish Process")


def ShuffleAndSeparateToTrainTestSet(directory, input_img_arg="img", input_txt_arg='txt', input_test_ratio=0.1):

    ROOT_DIR = directory
    IMG_DIR = os.path.join(ROOT_DIR, input_img_arg)
    print("Using default IMG_DIR = {}".format(IMG_DIR))

    txt_DIR = os.path.join(ROOT_DIR, input_txt_arg)
    print("Using default txt_DIR = {}".format(txt_DIR))

    print("Using default input_test_ratio = {}".format(input_test_ratio))

    IMG_DIR=os.path.normpath(IMG_DIR)
    txt_DIR=os.path.normpath(txt_DIR)
    PGM_DIR = IMG_DIR.replace('img', 'pgm')
    WAR_DIR = IMG_DIR.replace('img', 'war')
    TEST_DATASET_DIR = os.path.join(ROOT_DIR,"test")
    TRAIN_DATASET_DIR = os.path.join(ROOT_DIR, "train")

    ## Check if the Corresponding Directory exists 
    if not os.path.exists(IMG_DIR):
        raise ValueError("[Error] Directory named {} is not found. Please Check".format(os.path.basename(IMG_DIR)))

    if not os.path.exists(txt_DIR):
        raise ValueError("[Error] Directory named {} is not found. Please Check".format(os.path.basename(txt_DIR)))

    if not os.path.exists(TEST_DATASET_DIR):
        os.makedirs(TEST_DATASET_DIR)

    if not os.path.exists(TRAIN_DATASET_DIR):
        os.makedirs(TRAIN_DATASET_DIR)

    # Get file list
    txt_FILE_LIST = [f for f in glob.glob(txt_DIR + "/*.txt", recursive=True)]

    ## Check if the txt dir is empty
    if not (txt_FILE_LIST):
        raise ValueError("[Error] Directory named {} has no .txt file. Please Check".format(os.path.basename(txt_DIR)))
    print("[Processing] {} files are found in directory named {}"\
        .format(len(txt_FILE_LIST), os.path.basename(txt_DIR)))

    ## Shuffle the txt List
    random.shuffle(txt_FILE_LIST)
    test_file_num = round(float(input_test_ratio) * len(txt_FILE_LIST))

    train_set = []
    test_set = []

    ## Split the dataset into two part: train_set & test_set
    for i in range(test_file_num):
        elem = txt_FILE_LIST.pop()
        test_set.append(elem)

    train_set = [f for f in txt_FILE_LIST]


    # Overwrite file
    with open (os.path.join(ROOT_DIR, "train_labels.csv"), "w") as fptr:
        fptr.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
    with open (os.path.join(ROOT_DIR, "test_labels.csv"), "w") as fptr:
        fptr.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
    print("[Processing] {} files in test set".format(len(test_set)))
    print("[Processing] {} files in train_set".format(len(train_set)))

    print("[Processing] Copying of test dataset")
    pbar = tqdm(test_set)
    
    for iItems,  f in enumerate(pbar):
        file_basename = os.path.basename(f).replace('.txt','')
        
        for iImg in range(8):

            ext = "_" + str(iImg)
            try:
                shutil.copy(os.path.join(IMG_DIR, file_basename + ext + ".jpg"),                        TEST_DATASET_DIR)
                shutil.copyfile(f,                                                                      os.path.join(TEST_DATASET_DIR, file_basename + ext +'.txt'))
                shutil.copyfile(os.path.join(PGM_DIR, os.path.basename(f).replace(".txt", "_0.pgm")),   os.path.join(TEST_DATASET_DIR, file_basename + ext +'.pgm') )
                shutil.copy(os.path.join(WAR_DIR, os.path.basename(f).replace(".txt", "_0.war")),       os.path.join(TEST_DATASET_DIR, file_basename + ext +'.war') )
            except:
                print("Error copying files:\n {}".format(f))
                sys.exit()

            # Create csv
            boxes = read_as_list(os.path.join(TEST_DATASET_DIR, file_basename + ext +'.txt'))
            WriteListToFile(boxes, os.path.join(ROOT_DIR, "test_labels.csv"), "a")
        
        pbar.set_description("Items: %d" %iItems)


    print("[Processing] Copying of train dataset")
    pbar = tqdm(train_set)
    for iItems,  f in enumerate(pbar):
        file_basename = os.path.basename(f).replace('.txt','')
        
        for iImg in range(8):
            ext = "_" + str(iImg)
            try:
                shutil.copy(os.path.join(IMG_DIR, file_basename + ext + ".jpg"),                        TRAIN_DATASET_DIR)
                shutil.copyfile(f,                                                                      os.path.join(TRAIN_DATASET_DIR, file_basename + ext +'.txt'))
                shutil.copyfile(os.path.join(PGM_DIR, os.path.basename(f).replace(".txt", "_0.pgm")),   os.path.join(TRAIN_DATASET_DIR, file_basename + ext +'.pgm') )
                shutil.copy(os.path.join(WAR_DIR, os.path.basename(f).replace(".txt", "_0.war")),       os.path.join(TRAIN_DATASET_DIR, file_basename + ext +'.war') )
            except:
                print("Error copying files:\n {}".format(f))
                sys.exit()

            # Create csv
            boxes = read_as_list(os.path.join(TRAIN_DATASET_DIR, file_basename + ext +'.txt'))
            WriteListToFile(boxes, os.path.join(ROOT_DIR, "train_labels.csv"), "a")
        pbar.set_description("Items: %d" %iItems)

    print("[Processing] Finish Process")


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


def SingleImageShuffleAndSeparateToTrainTestSet(directory, input_img_arg="img", input_txt_arg='txt', input_test_ratio=0.1):

    ROOT_DIR = directory
    IMG_DIR = os.path.join(ROOT_DIR, input_img_arg)
    print("Using default IMG_DIR = {}".format(IMG_DIR))

    txt_DIR = os.path.join(ROOT_DIR, input_txt_arg)
    print("Using default txt_DIR = {}".format(txt_DIR))

    print("Using default input_test_ratio = {}".format(input_test_ratio))

    IMG_DIR=os.path.normpath(IMG_DIR)
    txt_DIR=os.path.normpath(txt_DIR)
    TEST_DATASET_DIR = os.path.join(ROOT_DIR,"test")
    TRAIN_DATASET_DIR = os.path.join(ROOT_DIR, "train")

    ## Check if the Corresponding Directory exists 
    if not os.path.exists(IMG_DIR):
        raise ValueError("[Error] Directory named {} is not found. Please Check".format(os.path.basename(IMG_DIR)))

    if not os.path.exists(txt_DIR):
        raise ValueError("[Error] Directory named {} is not found. Please Check".format(os.path.basename(txt_DIR)))

    if not os.path.exists(TEST_DATASET_DIR):
        os.makedirs(TEST_DATASET_DIR)

    if not os.path.exists(TRAIN_DATASET_DIR):
        os.makedirs(TRAIN_DATASET_DIR)

    # Get file list
    txt_FILE_LIST = [f for f in glob.glob(txt_DIR + "/*.txt", recursive=True)]

    ## Check if the txt dir is empty
    if not (txt_FILE_LIST):
        raise ValueError("[Error] Directory named {} has no .txt file. Please Check".format(os.path.basename(txt_DIR)))
    print("[Processing] {} files are found in directory named {}"\
        .format(len(txt_FILE_LIST), os.path.basename(txt_DIR)))

    ## Shuffle the txt List
    random.shuffle(txt_FILE_LIST)
    test_file_num = round(float(input_test_ratio) * len(txt_FILE_LIST))

    train_set = []
    test_set = []

    ## Split the dataset into two part: train_set & test_set
    for i in range(test_file_num):
        elem = txt_FILE_LIST.pop()
        test_set.append(elem)

    train_set = [f for f in txt_FILE_LIST]


    # Overwrite file
    with open (os.path.join(ROOT_DIR, "train_labels.csv"), "w") as fptr:
        fptr.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
    with open (os.path.join(ROOT_DIR, "test_labels.csv"), "w") as fptr:
        fptr.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
    print("[Processing] {} files in test set".format(len(test_set)))
    print("[Processing] {} files in train_set".format(len(train_set)))

    print("[Processing] Copying of test dataset")
    pbar = tqdm(test_set)
    
    for iItems,  f in enumerate(pbar):
        file_basename = os.path.basename(f).replace('.txt','')
        ext = "" 
        try:
            shutil.copy(os.path.join(IMG_DIR, file_basename + ext + ".jpg"),                        TEST_DATASET_DIR)
            shutil.copyfile(f,                                                                      os.path.join(TEST_DATASET_DIR, file_basename + ext +'.txt'))
        except:
            print("Error copying files:\n {}".format(f))
            sys.exit()

        # Create csv
        boxes = read_as_list(os.path.join(TEST_DATASET_DIR, file_basename + ext +'.txt'))
        WriteListToFile(boxes, os.path.join(ROOT_DIR, "test_labels.csv"), "a")
        
        pbar.set_description("Items: %d" %iItems)

    print("[Processing] Copying of train dataset")
    pbar = tqdm(train_set)
    for iItems,  f in enumerate(pbar):
        file_basename = os.path.basename(f).replace('.txt','')        
        ext = ""
        try:
            shutil.copy(os.path.join(IMG_DIR, file_basename + ext + ".jpg"),                        TRAIN_DATASET_DIR)
            shutil.copyfile(f,                                                                      os.path.join(TRAIN_DATASET_DIR, file_basename + ext +'.txt'))
        except:
            print("Error copying files:\n {}".format(f))
            sys.exit()

        # Create csv
        boxes = read_as_list(os.path.join(TRAIN_DATASET_DIR, file_basename + ext +'.txt'))
        WriteListToFile(boxes, os.path.join(ROOT_DIR, "train_labels.csv"), "a")
        pbar.set_description("Items: %d" %iItems)

    print("[Processing] Finish Process")


def main():
    # ShuffleAndSeparate()
    ROOT_DIR = "D:/FZ_WS/JyNB/TF_Research_Api_LD_2_0/research/object_detection/images/H_Dataset_00"
    ShuffleAndSeparateForEightChannel(ROOT_DIR)


if __name__ == '__main__':
    main()