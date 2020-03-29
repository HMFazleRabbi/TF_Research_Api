import os
import sys
import pandas as pd


def filter_body_xydxdy():
        
    ROOT_DIR = os.getcwd()

    IMG_DIR_PATH = os.path.join(ROOT_DIR, "img")
    TXT_DIR_PATH = os.path.join(ROOT_DIR, "text")

    IMG_FILES_PATH = [os.path.join(IMG_DIR_PATH,f) for f in os.listdir(IMG_DIR_PATH)]
    TXT_FILES_PATH = [os.path.join(TXT_DIR_PATH,f) for f in os.listdir(TXT_DIR_PATH)]


    for f in TXT_FILES_PATH:

        try:
            info_df = pd.read_csv(f, delimiter=" ", \
                            names =['x','y','dx','dy','label'], \
                            dtype ={'x': int, 'y':int, 'dx':int, 'dy':int, 'label': str})
        except:
            print("ERROR: Image: %s failed"%(f))
            sys.exit()

        found = any(info_df.label == "Body")
        
        if not found:
            print("{} not found".format(f))
            os.remove(f)
            os.remove(f.replace("text","img").replace(".txt",".jpg"))

def filter_body_x1y1x2y2():
    ROOT_DIR = os.getcwd()

    IMG_DIR_PATH = os.path.join(ROOT_DIR, "img")
    TXT_DIR_PATH = os.path.join(ROOT_DIR, "text")

    IMG_FILES_PATH = [os.path.join(IMG_DIR_PATH,f) for f in os.listdir(IMG_DIR_PATH)]
    TXT_FILES_PATH = [os.path.join(TXT_DIR_PATH,f) for f in os.listdir(TXT_DIR_PATH)]

def main():
    filter_body_xydxdy()

if if __name__ == "__main__":
    main()
    pass