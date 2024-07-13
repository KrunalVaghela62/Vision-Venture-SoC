#!/usr/bin/env python3

"""
6.101 Lab:
Image Processing
"""

import math

from PIL import Image

# NO ADDITIONAL IMPORTS ALLOWED!


def get_pixel(image, row, col,boundary_behavior):
    """
    Given row index and column index, this function
    returns the pixel from image["pixels"] 
    based on the boundary behavior
    """
    if 0<=row<image["height"] and 0<=col<image["width"]:
     return image["pixels"][image["width"]*row+col]
    else:
        if boundary_behavior=="zero":
            return 0
        if boundary_behavior=="extend":
            if row in range(0,image["height"])and col<0:
                return image["pixels"][image["width"]*row]
            if row in range(0,image["height"])and col>=image["width"]:
                return image["pixels"][(image["width"]*row)+image["width"]-1]
            if col in range(0,image["width"])and row<0:
                return image["pixels"][col]
            if col in range(0,image["width"])and row>=image["height"]:
                return image["pixels"][image["width"]*(image["height"]-1)+col]
            if row<0 and col<0:
                return image["pixels"][0]
            if row<0 and col>=image["width"]:
                return image["pixels"][image["width"]-1]
            if row>=image["height"] and col<0:
                return image["pixels"][image["width"]*(image["height"]-1)]
            if row>=image["height"] and col>=image["width"]:
                return image["pixels"][image["width"]*(image["height"]-1)+image["width"]-1]
        if boundary_behavior=="wrap":
            if row in range(0,image["height"])and col<0:
                if (abs(col)%image["width"])==0:
                    return image["pixels"][image["width"]*row+(abs(col)%image["width"])]
                else:
                    return image["pixels"][image["width"]*row+(image["width"]-(abs(col)%image["width"]))]
            if  row in range(0,image["height"] )and col>=image["width"]:
                return  image["pixels"][image["width"]*row+(abs(col)%image["width"])]
            if col in range(0,image["width"])and row<0:
                if (abs(row)%image["height"])==0:
                    return image["pixels"][image["width"]*(abs(row)%image["height"])+col]
                else:
                    return image["pixels"][image["width"]*(image["height"]-(abs(row)%image["height"]))+col]
            if  col in range(0,image["width"]) and row>=image["height"]:
                return  image["pixels"][image["width"]*(abs(row)%image["height"])+col]
            if row<0 and col<0:
                if abs(row)%image["height"]==0 and abs(col)%image["width"]!=0:
                    return image["pixels"][(image["width"]-(abs(col)%image["width"]))]
                if abs(row)%image["height"]!=0 and abs(col)%image["width"]==0:
                    return image["pixels"][image["width"]*(image["height"]-(abs(row)%image["height"]))]
                if abs(row)%image["height"]==0 and abs(col)%image["width"]==0:
                    return image["pixels"][0]
                if abs(row)%image["height"]!=0 and abs(col)%image["width"]!=0:
                    return image["pixels"][image["width"]*((image["height"]-(abs(row)%image["height"])))+(image["width"]-(abs(col)%image["width"]))]
            if row>=image["height"] and col<0:
                if abs(row)%image["height"]==0 and abs(col)%image["width"]!=0:
                    return image["pixels"][(image["width"]-(abs(col)%image["width"]))]
                if abs(row)%image["height"]!=0 and abs(col)%image["width"]==0:
                    return image["pixels"][image["width"]*((abs(row)%image["height"]))]
                if abs(row)%image["height"]==0 and abs(col)%image["width"]==0:
                    return image["pixels"][0]
                if abs(row)%image["height"]!=0 and abs(col)%image["width"]!=0:
                    return image["pixels"][image["width"]*((abs(row)%image["height"]))+(image["width"]-(abs(col)%image["width"]))]
            if col>=image["width"] and row<0:
                if abs(row)%image["height"]==0 and abs(col)%image["width"]!=0:
                    return image["pixels"][((abs(col)%image["width"]))]
                if abs(row)%image["height"]!=0 and abs(col)%image["width"]==0:
                    return image["pixels"][image["width"]*(image["height"]-(abs(row)%image["height"]))]
                if abs(row)%image["height"]==0 and abs(col)%image["width"]==0:
                    return image["pixels"][0]
                if abs(row)%image["height"]!=0 and abs(col)%image["width"]!=0:
                    return image["pixels"][image["width"]*((image["height"]-(abs(row)%image["height"])))+((abs(col)%image["width"]))]
            if col>=image["width"] and row>=image["height"]:
                    return image["pixels"][image["width"]*(((abs(row)%image["height"])))+((abs(col)%image["width"]))]
def set_pixel(image, row, col, color):
    image["pixels"][row*image["width"] + col] = color


def apply_per_pixel(image, func):
    """
    Apply the given function to each pixel in an image and return the resulting image.

    Parameters:
    image (dict): A dictionary representing an image, with keys:
                    - "height" (int): The height of the image.
                    - "width" (int): The width of the image.
                    - "pixels" (list): A list of pixel values.
    func (callable): A function that takes a single pixel value and returns a new pixel value.

    Returns:
    dict: A new image dictionary with the same dimensions as the input image, but with each
            pixel value modified by the given function.

    """
    new_image={
    "height":image["height"],
    "width":image["width"],
    "pixels":[func(x) for x in image["pixels"]],
    
} 
    return new_image

    raise NotImplementedError

    


def inverted(image):
    return apply_per_pixel(image, lambda color: 255-color)


# HELPER FUNCTIONS

def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    """
    if kernel["height"]!=kernel["width"]:
        raise ValueError("kernel must be square")
    if (kernel["height"]%2)==0 or kernel["width"]%2==0:
        raise ValueError("kerenl must be of odd dimensions")
    n=kernel["height"]
   
    output_image={
        "height":image["height"],
        "width": image["width"],
        "pixels":[0]*(image["width"]*image["height"])
    
    }
    if boundary_behavior=="zero":
        
        for i in range(0,image["height"]):
            for j in range(0,image["width"]):
                sum=0
                for l in range(i-int(((n-1)/2)),i+int(((n-1)/2))+1):
                    for m in range(j-int(((n-1)/2)),j+int(((n-1)/2))+1):
                        pixel_value=get_pixel(image,l,m,"zero")
                        sum += pixel_value*kernel["pixels"][kernel["width"]*(l-(i-int(((n-1)/2))))+m-int((j-((n-1)/2)))]
                output_image["pixels"][image["width"] * i + j] = sum
    if boundary_behavior=="extend":
        
        for i in range(0,image["height"]):
            for j in range(0,image["width"]):
                sum=0
                for l in range(i-int(((n-1)/2)),i+int(((n-1)/2))+1):
                    for m in range(j-int(((n-1)/2)),j+int(((n-1)/2))+1):
                        pixel_value=get_pixel(image,l,m,"extend")
                        sum += pixel_value*kernel["pixels"][kernel["width"]*(l-(i-int(((n-1)/2))))+m-int((j-((n-1)/2)))]
                output_image["pixels"][image["width"] * i + j] = sum
    if boundary_behavior=="wrap":
        
        for i in range(0,image["height"]):
            for j in range(0,image["width"]):
                sum=0
                for l in range(i-int(((n-1)/2)),i+int(((n-1)/2))+1):
                    for m in range(j-int(((n-1)/2)),j+int(((n-1)/2))+1):
                        pixel_value=get_pixel(image,l,m,"wrap")
                        sum += pixel_value*kernel["pixels"][kernel["width"]*(l-(i-int(((n-1)/2))))+m-int((j-((n-1)/2)))]
                output_image["pixels"][image["width"] * i + j] = sum
        
        
                    
    return output_image      
    raise NotImplementedError                   
                
   
                    
  
            


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    image["pixels"]=[round(x) for x in image["pixels"]]
    for i in range(len(image["pixels"])):
        if image["pixels"][i]<0:
            image["pixels"][i]=0
        if image["pixels"][i]>255:
            image["pixels"][i]=255
            
    return image
    raise NotImplementedError


# FILTERS

def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)

    # then compute the correlation of the input image with that kernel

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    def create_box_blur_kernel(kernel_size): #creating a helper function for creating kernel
        kernel={
            "height":kernel_size,
            "width":kernel_size,
            "pixels":[1/((kernel_size)**2)]*((kernel_size)**2),
        }
        return kernel
    correlate(image, create_box_blur_kernel(kernel_size), "extend")
    output_blurred_image=round_and_clip_image(correlate(image, create_box_blur_kernel(kernel_size), "extend"))
    
    return output_blurred_image
def test_blurred_black_image(kernel_size):
    i={
        "height":6,
        "width":5,
        "pixels":[0]*(6*5),
    }
    return blurred(i, kernel_size)
    
    

def sharpened(image, n):
    """
    Return a new image which is sharper than the input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    Output list should be clipped.
    """
    def create_box_blur_kernel1(n): #creating a helper function for creating kernel
        kernel={
            "height":n,
            "width":n,
            "pixels":[1/(n**2)]*((n**2)),
        }
        return kernel
    c=correlate(image, create_box_blur_kernel1(n), "extend")
    output_sharpened_image={
        "height":image["height"],
        "width":image["width"],
        "pixels":[0]*(image["height"]*image["width"]),
    }
    for i in range(image["height"]):
        for j in range(image["width"]):
            output_sharpened_image["pixels"][image["width"]*i+j]=2*(get_pixel(image,i,j,"extend"))-c["pixels"][c["width"]*i+j]
    return round_and_clip_image(output_sharpened_image)
    

def edges(image):
    """
    Return a new image with all the edges distincly and clearly detectable.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    Output list should be clipped.
    """
    K1={
        "height":3,
        "width":3,
        "pixels":[-1,-2,-1,0,0,0,1,2,1],
    }
    K2={
        "height":3,
        "width":3,
        "pixels":[1,0,1,-2,0,2,-1,0,1],
    }
    o_1=correlate(image, K1, "extend")
    o_2=correlate(image, K2, "extend")
    edged_image={
        "height":image["height"],
        "width":image["width"],
        "pixels":[0]*(image["width"]*image["height"]),
    }
    for i in range(image["height"]):
        for j in range(image["width"]):
            edged_image["pixels"][image["width"]*i+j]=math.sqrt((get_pixel(o_1,i,j,"extend"))**2+(get_pixel(o_2,i,j,"extend"))**2)
    return(round_and_clip_image(edged_image))
    
    raise NotImplementedError

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image("test_images/cat.png")
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the "mode" parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    pass

i=load_greyscale_image(r"C:\Users\Krunal\Desktop\SOC week3\test_images\mushroom.png")
k={
    "height":3,
    "width":3,
    "pixels":[2.4,-0.5,5,-0.005,0.5,10,-8,0.25,-0.4],
}
i_1=round_and_clip_image(correlate(i,k,"extend"))
save_greyscale_image(i_1,r"C:\Users\Krunal\Desktop\SOC week3\mushroom.png",mode="PNG")
