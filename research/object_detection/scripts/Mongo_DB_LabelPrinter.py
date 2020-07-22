import pandas as pd
import os, glob, shutil
from tqdm import tqdm
import datetime
import cv2

from validation_library import FAULT_NAME
print("Setup Complete")

import pymongo
from pprint import pprint
import json
from bson.objectid import ObjectId
print("Mongo db load complete")



class Save_Aoi_Images():

    '''
    Author: Fazle
    Date: 20200522_1004
    Description: Save_images just print the whole rgb fov  along per refdes
    '''
    def save_fov_per_refdes (myboard_col, mylabel_col, myquery, output_dir, img_root):
        
        mydoc = myboard_col.find(myquery)
        print("Analysing records: ", (mydoc.count()))
        
        for row in mydoc:
            ObjectId = row['_id']
            bname=row['board_name']
            
            # Make dir
            out_board_dir = os.path.normpath(os.path.join(output_dir,bname))
            out_board_dir = os.path.normpath(os.path.join(output_dir,""))
            if not (os.path.isdir(out_board_dir)):
                os.makedirs(out_board_dir)
            
            # Path
            img_path=os.path.normpath(os.path.join(img_root, row['path']))
            new_imgname=out_board_dir +"/" + str(ObjectId)+".jpg"
            new_txtname=out_board_dir +"/" + str(ObjectId)+".txt"
            
            # Write txt
            # Pin and Body
            with open(new_txtname, "w") as f:
                f.write("{},{},{},{},PACKAGE___{}\n".format(row['search_area']['cx'] -10, row['search_area']['cy']-10, row['search_area']['cx'] +10, row['search_area']['cy']+10, row['package']))
                f.write("{},{},{},{},search_area\n".format(row['search_area']['x1'], row['search_area']['y1'], row['search_area']['x2'], row['search_area']['y2']))
                if FAULT_NAME['invalidbodydimensions'] in row['error_stack']:
                    f.write("{},{},{},{},body_dim\n".format(row['search_area']['x1']+10, row['search_area']['y1']+10, row['search_area']['x2']-10, row['search_area']['y2']-10))
                else:
                    f.write("{},{},{},{},body_dim\n".format(row['body_dim']['x1'], row['body_dim']['y1'], row['body_dim']['x2'], row['body_dim']['y2']))  
                    
                
                # Pin
                if (row['pins']['len'])>0:
                    for pin_row in row['pins']['dim_list']:
                        f.write("{},{},{},{},pin\n".format(pin_row['x1'], pin_row['y1'], pin_row['x2'], pin_row['y2']))


            # Move
            shutil.copy(img_path, out_board_dir)
            shutil.move(os.path.join(out_board_dir, os.path.basename(img_path)), new_imgname)
            document={
                "board_information_fkey": row['_id'],
                "last_modified":datetime.datetime.now(),
                
                "img_path": new_imgname,
                "txt_path": new_txtname
            }
            mylabel_col.insert_one(document)
            
            # Class files
            class_txt_path=out_board_dir +"/classes.txt"
            if not os.path.isfile(class_txt_path):
                with open(class_txt_path, "w") as f:
                    f.write("search_area\nbody_dim\npin")
                    pass

    '''
    Author: Fazle
    Date: 20200705_1732
    Description: Save only the component search area but exapand the the border by 10pix soo that
    we are able to have a bbox for the search area
    '''
    def save_searcharea_extended (myboard_col, mylabel_col, myquery, output_dir, img_root):
        
        mydoc = myboard_col.find(myquery)
        print("Analysing records: ", (mydoc.count()))
        
        for row in mydoc:
            ObjectId = row['_id']
            bname=row['board_name']
            
            # Make dir
            out_board_dir = os.path.normpath(os.path.join(output_dir,bname))
            out_board_dir = os.path.normpath(os.path.join(output_dir,""))
            if not (os.path.isdir(out_board_dir)):
                os.makedirs(out_board_dir)
            
            # Path
            img_path=os.path.normpath(os.path.join(img_root, row['path']))
            new_imgname=out_board_dir +"/" + str(ObjectId)+".jpg"
            new_txtname=out_board_dir +"/" + str(ObjectId)+".txt"
            
            # Write txt
            # Pin and Body
            with open(new_txtname, "w") as f:
                f.write("{},{},{},{},PACKAGE___{}\n".format(row['search_area']['cx'] -10, row['search_area']['cy']-10, row['search_area']['cx'] +10, row['search_area']['cy']+10, row['package']))
                f.write("{},{},{},{},search_area\n".format(row['search_area']['x1'], row['search_area']['y1'], row['search_area']['x2'], row['search_area']['y2']))
                if FAULT_NAME['invalidbodydimensions'] in row['error_stack']:
                    f.write("{},{},{},{},body_dim\n".format(row['search_area']['x1']+10, row['search_area']['y1']+10, row['search_area']['x2']-10, row['search_area']['y2']-10))
                else:
                    f.write("{},{},{},{},body_dim\n".format(row['body_dim']['x1'], row['body_dim']['y1'], row['body_dim']['x2'], row['body_dim']['y2']))  
                    
                
                # Pin
                if (row['pins']['len'])>0:
                    for pin_row in row['pins']['dim_list']:
                        f.write("{},{},{},{},pin\n".format(pin_row['x1'], pin_row['y1'], pin_row['x2'], pin_row['y2']))

            # Move
            shutil.copy(img_path, out_board_dir)
            shutil.move(os.path.join(out_board_dir, os.path.basename(img_path)), new_imgname)
            img = cv2.imread(new_imgname)
            img=img[abs(row['search_area']['y1']-10) : row['search_area']['y2']+10, abs(row['search_area']['x1']-10): row['search_area']['x2']+10]
            cv2.imwrite(new_imgname,img)
            
            document={
                "board_information_fkey": row['_id'],
                "last_modified":datetime.datetime.now(),
                
                "img_path": new_imgname,
                "txt_path": new_txtname
            }
            mylabel_col.insert_one(document)
            
            # Class files
            class_txt_path=out_board_dir +"/classes.txt"
            if not os.path.isfile(class_txt_path):
                with open(class_txt_path, "w") as f:
                    f.write("search_area\nbody_dim\npin")
                    pass
    
            
    '''
    Author: Fazle
    Date: 20200520_1217
    Description: To print the search area images for all channel as grayscale, so essentially all 
    9 channel if present. The object name will be postfix with the channel number.
    '''
    def save_searcharea(myboard_col, mylabel_col, myquery, output_dir, img_root, scheme=0):
        
        mydoc = myboard_col.find(myquery)
        print("Analysing records: ", (mydoc.count()))
        
        for row in mydoc:
            ObjectId = row['_id']
            bname=row['board_name']
            total_channels = row['tile_files_info']['total_channels']
            channel_list = row['tile_files_info']['channel_list']
            path_split = os.path.split(os.path.normpath(row['path']))
            tile_no = os.path.split(path_split[0])[1]


            # Make dir
            out_board_dir = os.path.normpath(os.path.join(output_dir,bname))
            out_board_dir = os.path.normpath(os.path.join(output_dir,""))
            if not (os.path.isdir(out_board_dir)):
                os.makedirs(out_board_dir)
            
            # Write txt
            # Pin and Body
            sa = row['search_area']
            new_txtname=out_board_dir +"/" + str(ObjectId)+".txt"
            with open(new_txtname, "w") as f:
                # Package Name
                f.write("{},{},{},{},PACKAGE___{}\n".format(row['search_area']['cx'] -10-sa['x1'], 
                                                            row['search_area']['cy']-10 -sa['y1'], 
                                                            row['search_area']['cx'] +10-sa['x1'], 
                                                            row['search_area']['cy']+10 -sa['y1'], 
                                                            row['package']))
                
                
                # Body
                if FAULT_NAME['invalidbodydimensions'] in row['error_stack']:
                    f.write("{},{},{},{},BODY\n".format(row['search_area']['x1']+10 -sa['x1'],
                                                        row['search_area']['y1']+10 -sa['y1'],
                                                        row['search_area']['x2']-10 -sa['x1'], 
                                                        row['search_area']['y2']-10 -sa['y1']))
                else:
                    f.write("{},{},{},{},BODY\n".format(row['body_dim']['x1'] -sa['x1'],
                                                        row['body_dim']['y1'] -sa['y1'],
                                                        row['body_dim']['x2'] -sa['x1'], 
                                                        row['body_dim']['y2'] -sa['y1']
                                                    ))  
                    
                
                # Pin
                if (row['pins']['len'])>0:
                    for pin_row in row['pins']['dim_list']:
                        f.write("{},{},{},{},PIN\n".format(pin_row['x1'] -sa['x1'], 
                                                        pin_row['y1'] -sa['y1'], 
                                                        pin_row['x2'] -sa['x1'], 
                                                        pin_row['y2'] -sa['y1']
                                                        ))
                        

            ###################################
            # Save
            ###################################
            if (scheme == 1):
                #Rgb images
                new_imgname=out_board_dir +"/" + str(ObjectId)+".jpg"
                
                # Move
                img_path = os.path.normpath(os.path.join(os.path.normpath(img_root), row['path']))
                img = cv2.imread(img_path)
                img = img[abs(sa['y1']) : abs(sa['y2']), abs(sa['x1']) : abs(sa['x2'])]
                cv2.imwrite(new_imgname, img)
                
                # Save to a logging database
                document={
                    "board_information_fkey": row['_id'],
                    "last_modified":datetime.datetime.now(),
                    
                    "img_path": new_imgname,
                    "txt_path": new_txtname
                }
                mylabel_col.insert_one(document)
            elif (scheme == 2):
                #Rgb images as seperate rgb channel in grayscale
                temp_txtname = new_txtname
                new_imgname=out_board_dir +"/" + str(ObjectId)+".jpg"
                name0=out_board_dir +"/" + str(ObjectId)+"_0.jpg"
                name1=out_board_dir +"/" + str(ObjectId)+"_1.jpg"
                name2=out_board_dir +"/" + str(ObjectId)+"_2.jpg"
                
                
                # Move
                img_path = os.path.normpath(os.path.join(os.path.normpath(img_root), row['path']))
                img = cv2.imread(img_path)
                img = img[abs(sa['y1']) : abs(sa['y2']), abs(sa['x1']) : abs(sa['x2'])]
                
                # Channels
                cv2.imwrite(name0, img[:,:,0])
                cv2.imwrite(name1, img[:,:,1])
                cv2.imwrite(name2, img[:,:,2])
                
                # Text
                shutil.copy(temp_txtname, name0.replace(".jpg", ".txt"))
                shutil.copy(temp_txtname, name1.replace(".jpg", ".txt"))
                shutil.copy(temp_txtname, name2.replace(".jpg", ".txt"))
                os.remove(temp_txtname)
                
                # Save to a logging database
                document={
                    "board_information_fkey": row['_id'],
                    "last_modified":datetime.datetime.now(),
                    "img_path": new_imgname,
                    "txt_path": new_txtname
                }
                mylabel_col.insert_one(document)
            elif (scheme == 3):
                # Individual tile images 
                # Loop over channels
                temp_txtname = new_txtname
                for channel in channel_list:

                    # Path
                    new_imgname=out_board_dir +"/" + str(ObjectId)+"_{}.jpg".format(channel)
                    new_txtname=out_board_dir +"/" + str(ObjectId)+"_{}.txt".format(channel)

                    P = path_split[1].replace("_RGB.jpg", "_{}_{}.jpg".format(tile_no, channel))
                    img_path=os.path.normpath(os.path.join(os.path.normpath(img_root), path_split[0], os.path.normpath(P)))


                    # Move
                    shutil.copy(temp_txtname, new_txtname)
                    img = cv2.imread(img_path)
                    img=img[abs(sa['y1']) : abs(sa['y2']), abs(sa['x1']) : abs(sa['x2'])]
                    cv2.imwrite(new_imgname, img)

                    # Save to a logging database
                    document={
                        "board_information_fkey": row['_id'],
                        "last_modified":datetime.datetime.now(),
                        "img_path": new_imgname,
                        "txt_path": new_txtname
                    }
                    mylabel_col.insert_one(document)
                os.remove(temp_txtname)
            else:
                print("ERROR: Invalid Scheme!")
                return
            
        
    '''
    Author: Fazle
    Date: 20200522_0941
    Description: Special version of save_images_searcharea where the pins are of different type.
    '''
    def save_searcharea_withpintype(myboard_col, mylabel_col, myquery, output_dir, img_root, scheme=0):
        
        mydoc = myboard_col.find(myquery)
        print("Analysing records: ", (mydoc.count()))
        
        for row in mydoc:
            ObjectId = row['_id']
            bname=row['board_name']
            total_channels = row['tile_files_info']['total_channels']
            channel_list = row['tile_files_info']['channel_list']
            path_split = os.path.split(os.path.normpath(row['path']))
            tile_no = os.path.split(path_split[0])[1]


            # Make dir
            out_board_dir = os.path.normpath(os.path.join(output_dir,bname))
            out_board_dir = os.path.normpath(os.path.join(output_dir,""))
            if not (os.path.isdir(out_board_dir)):
                os.makedirs(out_board_dir)
            
            # Write txt
            # Pin and Body
            sa = row['search_area']
            new_txtname=out_board_dir +"/" + str(ObjectId)+".txt"
            with open(new_txtname, "w") as f:
                if FAULT_NAME['invalidbodydimensions'] in row['error_stack']:
                    f.write("{},{},{},{},BODY\n".format(row['search_area']['x1']+10 -sa['x1'],
                                                        row['search_area']['y1']+10 -sa['y1'],
                                                        row['search_area']['x2']-10 -sa['x1'], 
                                                        row['search_area']['y2']-10 -sa['y1']))
                else:
                    f.write("{},{},{},{},BODY\n".format(row['body_dim']['x1'] -sa['x1'],
                                                        row['body_dim']['y1'] -sa['y1'],
                                                        row['body_dim']['x2'] -sa['x1'], 
                                                        row['body_dim']['y2'] -sa['y1']
                                                    ))  
                    
                
                # Pin
                if (row['pins']['len'])>0:
                    for pin_row in row['pins']['dim_list']:
                        f.write("{},{},{},{},{}\n".format(pin_row['x1'] -sa['x1'], 
                                                        pin_row['y1'] -sa['y1'], 
                                                        pin_row['x2'] -sa['x1'], 
                                                        pin_row['y2'] -sa['y1'],
                                                        package_pin_map[row['package']].strip().upper()
                                                        ))
            ###################################
            # Save
            ###################################
            if (scheme == 1):
                #Rgb images
                new_imgname=out_board_dir +"/" + str(ObjectId)+".jpg"
                
                # Move
                img_path = os.path.normpath(os.path.join(os.path.normpath(img_root), row['path']))
                img = cv2.imread(img_path)
                img = img[abs(sa['y1']) : abs(sa['y2']), abs(sa['x1']) : abs(sa['x2'])]
                cv2.imwrite(new_imgname, img)
                
                # Save to a logging database
                document={
                    "board_information_fkey": row['_id'],
                    "last_modified":datetime.datetime.now().strftime("%x"),
                    "last_modified_hour":datetime.datetime.now().strftime("%H"),
                    "img_path": new_imgname,
                    "txt_path": new_txtname
                }
                mylabel_col.insert_one(document)
            elif (scheme == 2):
                #Rgb images as seperate rgb channel in grayscale
                temp_txtname = new_txtname
                new_imgname=out_board_dir +"/" + str(ObjectId)+".jpg"
                name0=out_board_dir +"/" + str(ObjectId)+"_0.jpg"
                name1=out_board_dir +"/" + str(ObjectId)+"_1.jpg"
                name2=out_board_dir +"/" + str(ObjectId)+"_2.jpg"
                
                
                # Move
                img_path = os.path.normpath(os.path.join(os.path.normpath(img_root), row['path']))
                img = cv2.imread(img_path)
                img = img[abs(sa['y1']) : abs(sa['y2']), abs(sa['x1']) : abs(sa['x2'])]
                
                # Channels
                cv2.imwrite(name0, img[:,:,0])
                cv2.imwrite(name1, img[:,:,1])
                cv2.imwrite(name2, img[:,:,2])
                
                # Text
                shutil.copy(temp_txtname, name0.replace(".jpg", ".txt"))
                shutil.copy(temp_txtname, name1.replace(".jpg", ".txt"))
                shutil.copy(temp_txtname, name2.replace(".jpg", ".txt"))
                os.remove(temp_txtname)
                
                # Save to a logging database
                document={
                    "board_information_fkey": row['_id'],
                    "last_modified":datetime.datetime.now().strftime("%x"),
                    "last_modified_hour":datetime.datetime.now().strftime("%H"),
                    "img_path": new_imgname,
                    "txt_path": new_txtname
                }
                mylabel_col.insert_one(document)
            elif (scheme == 3):
                # Individual tile images 
                # Loop over channels
                temp_txtname = new_txtname
                for channel in channel_list:

                    # Path
                    new_imgname=out_board_dir +"/" + str(ObjectId)+"_{}.jpg".format(channel)
                    new_txtname=out_board_dir +"/" + str(ObjectId)+"_{}.txt".format(channel)

                    P = path_split[1].replace("_RGB.jpg", "_{}_{}.jpg".format(tile_no, channel))
                    img_path=os.path.normpath(os.path.join(os.path.normpath(img_root), path_split[0], os.path.normpath(P)))


                    # Move
                    shutil.copy(temp_txtname, new_txtname)
                    img = cv2.imread(img_path)
                    img=img[abs(sa['y1']) : abs(sa['y2']), abs(sa['x1']) : abs(sa['x2'])]
                    cv2.imwrite(new_imgname, img)

                    # Save to a logging database
                    document={
                        "board_information_fkey": row['_id'],
                        "last_modified":datetime.datetime.now().strftime("%x"),
                        "last_modified_hour":datetime.datetime.now().strftime("%H"),
                        "img_path": new_imgname,
                        "txt_path": new_txtname
                    }
                    mylabel_col.insert_one(document)
                os.remove(temp_txtname)
            else:
                print("ERROR: Invalid Scheme!")
                return                       
                
        