import pandas as pd
import os, glob, shutil
import seaborn as sns

FAULT_NAME = {
    "ok": "OK",
    "badjoint": "BADJOINT",
    "billboard": "BILLBOARD",
    "bridging": "BRIDGING",
    "coplanarity": "COPLANARITY",
    "damaged": "DAMAGED",
    "extra": "EXTRA",
    "lifted": "LIFTED",
    "missing": "MISSING",
    "ocv": "OCV",
    "offset": "OFFSET",
    "paste": "PASTE",
    "polarity": "POLARITY",
    "skew": "SKEW",
    "pendingverification":"PENDING VERIFICATION",
    "invalidpindimensions": "INVALID PIN DIMENSIONS",
    "missingpins": "MISSING PINS",
    "mismatchpinarraylength": "MISMATCH PIN ARRAY LENGTH",
    "invalidbodydimensions": "INVALID BODY DIMENSIONS",
    "mismatchbodycenter": "MISMATCH BODY CENTER",
    "mismatchbodywidth": "MISMATCH BODY WIDTH",
    "mismatchbodyheight": "MISMATCH BODY HEIGHT",
    "outofboundsearcharea": "OUTOFBOUND SEARCHAREA",
    "mismatchsearchareacenter": "MISMATCH SEARCH AREA CENTER",
    "invalidnegativecoordinates": "INVALID NEGATIVE COORDINATES"


}


# *************************************************************
#   Author       : HM Fazle Rabbi
#   Description  : Validation invalid coordinates in width and height
#   Date Modified: 
#   Copyright © 2000, MV Technology Ltd. All rights reserved.
# *************************************************************
def validate_wh(x1,y1, x2,y2, w, h):
    if ((x2-x1)!=w):
        return False, FAULT_NAME['mismatchbodywidth']
    if ((y2-y1)!=h):
        return False, FAULT_NAME['mismatchbodyheight']
    return True, FAULT_NAME["ok"]

# *************************************************************
#   Author       : HM Fazle Rabbi
#   Description  : Validation invalid coordinates in x1, y1...
#   Date Modified: 
#   Copyright © 2000, MV Technology Ltd. All rights reserved.
# *************************************************************
def validate_x1y1x2y2 (x1,x2,y1,y2, xmin, xmax, ymin, ymax, msg):
    success = True

    if ((x1 > xmax) or (x2 > xmax)):
        success = False
#         print("Failed: ((x1 > xmax) or (x2 > xmax))")
    if ((x1 < xmin) or (x2 < xmin)):
        success = False
        print("Failed: ((x1 < xmin) or (x2 < xmin))")

    if ((y1 > ymax) or (y2 > ymax)):
        success = False
        print("Failed: ((y1 > ymax) or (y2 > ymax))")
    if ((y1 < ymin) or (y2 < ymin)):
        success = False
        print("Failed: ((y1 < ymin) or (y2 < ymin))")

    if ((x2 - x1) <= 1 ):
        success = False
        print("Failed: (x2 <= x1)")
    if ((y2 - y1) <= 1):
        success = False
        print("Failed: (y2 <= y1)")

    if not success:
        print("Invalid coordinate:\n\t{} {} {} {} \n\t{} {} {} {}".format(x1,x2,y1,y2, xmin, xmax, ymin, ymax))
        print(msg)
        pass
    return success

# *************************************************************
#   Author       : HM Fazle Rabbi
#   Description  : Validation invalid coordinates in x1, y1...
#   Date Modified: 
#   Copyright © 2000, MV Technology Ltd. All rights reserved.
# *************************************************************
def validate_cxcy (cx, cy, xmin, xmax, ymin, ymax, msg):
    success = True
    if ((cx > xmax) or (cy > ymax)) :
        success = False
#         print("Failed: (cx > xmax) or (cy > ymax) ")
    if ((cx < xmin) or (cy < ymin)):
        success = False
#         print("Failed:  ((cx < xmin) or (cy < ymin))")
        if not success:
#            print(msg)
            pass
    return success

def validate_searcharea(document):
   
    # Negative coordinate
    if (document['search_area']['x2'] <3) or (document['search_area']['y2']<3) or (document['search_area']['x1'] <1) or (document['search_area']['y1']<1) :   
            return False, FAULT_NAME["invalidnegativecoordinates"] 
    
    # Flipped coordinate
    if (document['search_area']['x1'] >document['search_area']['x2']) or (document['search_area']['y1'] >document['search_area']['y2']) :
        return False, FAULT_NAME["outofboundsearcharea"] 


    # Center Validation
    if not (validate_cxcy (     
        cx=document['search_area']['cx'],
        cy=document['search_area']['cy'],
        xmin=document['search_area']['x1'],
        xmax=document['search_area']['x2'],
        ymin=document['search_area']['y1'],
        ymax=document['search_area']['y2'], 
        msg="[ERROR]: Invalid search_area center location!")):
        return False, FAULT_NAME["mismatchsearchareacenter"] 



    # Success
    return True, FAULT_NAME["ok"]

def validate_bodydimensions(document):
    if not (validate_x1y1x2y2 ( x1=document['body_dim']['x1'],
                                x2=document['body_dim']['x2'],
                                y1=document['body_dim']['y1'],
                                y2=document['body_dim']['y2'],
                                
                                xmin=document['search_area']['x1'],
                                xmax=document['search_area']['x2'],
                                ymin=document['search_area']['y1'],
                                ymax=document['search_area']['y2'],
                                msg="[ERROR]: Invalid body dimension or location!")):
        return False, FAULT_NAME["invalidbodydimensions"]   

    if not (validate_cxcy (     cx=document['body_dim']['cx'],
                                cy=document['body_dim']['cy'],
                                xmin=document['body_dim']['x1'],
                                xmax=document['body_dim']['x2'],
                                ymin=document['body_dim']['y1'],
                                ymax=document['body_dim']['y2'], 
                                msg="[ERROR]: Invalid body center location!")):
        return False, FAULT_NAME["mismatchbodycenter"] 


    # Center Validation
    success, error = validate_wh(x1=document['body_dim']['x1'],
                                y1=document['body_dim']['y1'],
                                x2=document['body_dim']['x2'],
                                y2=document['body_dim']['y2'],
                                w=document['body_dim']['width'],
                                h=document['body_dim']['height'])
    if not (success):
        return False, error

    # Success
    return True, FAULT_NAME["ok"]


def validate_pindimensions(document):
    #Length
    length = document['pins']['len']
    pins = document['pins']['dim_list']
    if(len(pins) != length):
        return False, FAULT_NAME["mismatchpinarraylength"]

    # Pin
    for pin in pins:
        if not (validate_x1y1x2y2 (
            x1=pin["x1"],
            x2=pin["x2"],
            y1=pin["y1"],
            y2=pin["y2"], 
            xmin=document['search_area']['x1'],
            xmax=document['search_area']['x2'],
            ymin=document['search_area']['y1'],
            ymax=document['search_area']['y2'],
            msg="[ERROR]: Invalid pin dimension or location!")):
            return False, FAULT_NAME["invalidpindimensions"]
    return True, FAULT_NAME["ok"]