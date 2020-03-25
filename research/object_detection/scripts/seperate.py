import os
import shutil
import glob
import string

ROOT_DIR = os.getcwd()

IMG_RGB_CLEAN_DIR = os.path.join(ROOT_DIR, "img_clean")
TXT_RGB_CLEAN_DIR = os.path.join(ROOT_DIR, "text_clean")
IMG_ALL_UNCLEAN_DIR = os.path.join(ROOT_DIR, "other_all")

OUTPUT_DIR = os.path.join(ROOT_DIR, "other_clean")

IMG_RGB_CLEAN_LIST = [ f for f in glob.glob(IMG_RGB_CLEAN_DIR + '/*.jpg', recursive=True)]

for img_rgb in IMG_RGB_CLEAN_LIST:
    refdes = os.path.basename(img_rgb).replace(".jpg","").split("_")[-1]
    package_list = os.path.basename(img_rgb).replace(".jpg","").split("_")[:-1]
    package_name = ''.join(package_list)
    IMG_OTH_ALL_LIST = [ f for f in glob.glob(IMG_ALL_UNCLEAN_DIR +'/'+ package_name + '*'+ refdes + '.jpg', recursive=True)]
    TXT_OTH_ALL_LIST = [ f for f in glob.glob(IMG_ALL_UNCLEAN_DIR +'/'+ package_name + '*'+ refdes + '.txt', recursive=True)]
    for f in IMG_OTH_ALL_LIST:
        shutil.copy(f, OUTPUT_DIR)
    
    for f in TXT_OTH_ALL_LIST:
        
        original_file = img_rgb.replace("img_clean","text_clean").replace(".jpg",".txt")
        old_name = os.path.basename(original_file)
        shutil.copy(original_file, OUTPUT_DIR)
        try:
            os.rename(os.path.join(OUTPUT_DIR, old_name), os.path.join(OUTPUT_DIR, os.path.basename(f)))

        except:
            print("File already exist")

