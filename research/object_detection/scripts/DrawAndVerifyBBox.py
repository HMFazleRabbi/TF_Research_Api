import cv2
import sys
import pandas as pd
import numpy as np
import os

PROGRAM_NAME = sys.argv[0]
input_dir = sys.argv[1]
output_dir = sys.argv[2]

ROOT_DIR = os.getcwd()
INPUT_DIR_PATH = os.path.join(ROOT_DIR,input_dir)
OUTPUT_DIR_PATH = os.path.join(ROOT_DIR, output_dir)


def DrawAndVerifyBBox():
    print("START")
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1
    scaleFactor  = 0.02##CHANGE FOR DIFFERENT SIZE IN RELATION TO HEIGHT
    fontColor = (0,0,0)
    fontColorU = (0,255,255)
    fontColorL = (0,255,0)
    fontColorN  = (230,255,0)
    lineType = 2

    img_dir_key = "img"
    txt_dir_key = "text"
    out_dir_key = "tst"

    img_dir = os.path.join(INPUT_DIR_PATH, img_dir_key)
    txt_dir = os.path.join(INPUT_DIR_PATH, txt_dir_key)
    tst_dir = os.path.join(OUTPUT_DIR_PATH, out_dir_key)

    if not os.path.exists(tst_dir):
        os.makedirs(tst_dir)

    img_list = [ os.path.join(img_dir, f) for f in os.listdir(img_dir) ]

    for f in img_list:
        img = cv2.imread(f)
        txtfile = f.replace(img_dir_key,txt_dir_key).replace(".jpg",".txt")
        outimg = f.replace(img_dir_key,out_dir_key)
        
        try:
            data = pd.read_csv(txtfile, delimiter=" ", \
                            names =['x','y','dx','dy','label'], \
                            dtype ={'x': int, 'y':int, 'dx':int, 'dy':int, 'label': str})
        except:
            print("ERROR: Image: %s failed"%(f))
            print("Text: %s IMG: %s"%(outimg, txtfile))
            sys.exit()
            
            
        yOffsetText=img.shape[0]
        img = cv2.copyMakeBorder(img, top=0, bottom=int(img.shape[0]*1.20), left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=0 );

        for index, row in data.iterrows():
            p1=(int (row['x']),int (row['y']))
            p2=(int(row['x']+row['dx']),int(row['y']+row['dy']))
            cv2.rectangle(img, p1,p2, (0,255,0), 1)
            
            if row['label'].isupper() and not row['label'].isnumeric():
                fontColor = fontColorU
            elif row['label'].islower() and not row['label'].isnumeric():
                fontColor = fontColorL
            else:
                fontColor = fontColorN
            
            fontScale = max (row['dy'],row['dx'])*scaleFactor
            cv2.putText(img, row['label'], (row['x'],row['y']+row['dy']+yOffsetText),\
                        font, fontScale, fontColor, lineType)

        cv2.imwrite(outimg, img)
        
    print("DONE")

if __name__=='__main__':
	DrawAndVerifyBBox()
