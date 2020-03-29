import re
import numpy as np

def read_hmap(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    headerRegex = re.compile(r'#Warpage,(-?[0-9]\d*(\.\d+)),(-?[0-9]\d*(\.\d+)), (-?[0-9]\d*(\.\d+))*[\r\n]*')
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\sd*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)

    # Warpage compensation
    lines= header.split(b"\n")
    print(lines)
    for line in lines:
        if b"#Warpage" in line:
            values= line.split(b",")
            mx = np.float (values[1])
            my = np.float (values[2])
            c  = np.float (values[3])

    # z = np.array([[x*mx + y*my  + c for x in range(np.int(width))] for y in range(np.int(height))])
    z=0
    hmap = np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

    return header, z, hmap-z



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    _, _, image = read_hmap("D:/FZ_WS/JyNB/Yolo_LD/Tf_Yolov3/LD_Files/Output_ComponentByBoard/9608_t_1_90_r/FILTERED/9608_t_1_90_r_1d301_0.pgm", byteorder='<')
    # print(image.shape)
    np.savetxt('test1.txt', image, fmt='%f')

    # Color map
    fig = plt.figure()
    image[image < 0] = 0
    image[image > 1000] = 1000
    p = plt.imshow(image)
    plt.colorbar(p)
    plt.show()
