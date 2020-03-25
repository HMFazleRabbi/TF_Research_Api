from pascal_voc_writer import Writer
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm

print('start')

ROOT_DIR = os.getcwd()

BASE_DIR = os.path.join(ROOT_DIR)
txtfolder= os.path.join(ROOT_DIR, "text")
imgfolder= os.path.join(ROOT_DIR, "img")
xmlfolder= os.path.join(ROOT_DIR, "xml")

pbar = tqdm(os.listdir(txtfolder))
for iItems, txtfile in enumerate(pbar):
    txtpath=os.path.join(txtfolder, txtfile)
    imgpath=os.path.join(imgfolder, txtfile.replace(".txt", ".jpg"))

    img = Image.open(imgpath)
    width, height = img.size
    
    XMLwriter = Writer(imgpath, width, height)
    
    data = pd.read_csv(txtpath, delimiter=" ",\
                       names =['x','y','dx','dy','class'],\
                       dtype ={ 'x': int, 'y':int, 'dx':int,\
                               'dy':int, 'class': str})
 
    for ind, row in data.iterrows():
        xmin = row['x']
        ymin = row['y']
        xmax = xmin + row['dx']
        ymax = ymin + row['dy']
        label = row['class']
              
        XMLwriter.addObject(label, xmin, ymin, xmax, ymax)
        
    XMLwriter.save(os.path.join(xmlfolder, txtfile.replace('.txt', '.xml')))
    pbar.set_description("Items: %.2f" %iItems)

print('done')