import find_image
import cv2
import sys
from matplotlib import pyplot as plt


templates = {
     'pedestrian_crossing_sign': 'template1.jpg',
     'stop_sign': 'template3.jpg'
}
MIN_MATCH_COUNT = 8
i = 0
for key,value in templates.iteritems():
     template = cv2.imread(str(value), 0) # Original image - grayscale
     sample = cv2.imread(str(sys.argv[1]), 0) #Sample image - grayscale
     if find_image.find_Image(template,sample, MIN_MATCH_COUNT, key) == None:
         template = cv2.flip(template,1)
         if find_image.find_Image(template,sample, MIN_MATCH_COUNT, key) == None:
            if i == len(templates)-1:
                print "There is no matches for this image"
            else:
                i+=1





