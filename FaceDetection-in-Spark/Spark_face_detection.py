# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from timeit import default_timer as timer

# <codecell>

# Start the time
start = timer()

# <codecell>

from pyspark import SparkContext
from pyspark import SparkFiles

# <codecell>

sc = SparkContext()

# <codecell>

import cv2
import numpy as np
import pickle

# <codecell>

# Images directory
# Add here the path of the directory which has images.
# You can also add the HDFS directory path. Suppose your images are saved in HDFS in the directory '/user/maddy/my_images/'.
# Then same HDFS path can be given here.
img_dir = './my_images/'

# This is the path of the directory where the images will be stored after face is detected.
# After the face is detected in the image, we will draw a rectangle around the face in the image & store that image in the below directory.
rect_img_dir = './face_detected/'

# <codecell>

# Haar Cascade Classifier (from OpenCV library)
# This classifier will be used to detect front faces in the images.
# Give below the path of the classifier.
distCascade = "./haarcascade_frontalface_default.xml"

# <codecell>

# This adds the Cascade file on different nodes in Spark cluster.
# This is necessary if you run this spark code on muti-node spark cluster.
sc.addFile(distCascade)

# <codecell>

# Converting the images into RDD
images_RDD = sc.binaryFiles(img_dir)
# For more details about this function. You can do help(sc.binaryFiles)

# If you have large number of images to process (like a million) then the Spark will by default make a lot of partitions.
# To repartition your image data into less number of partitions, you can run below command & change the number of partitions to what you want.
#images_RDD = images_RDD.repartition(20000)

# Face Detection function
def face_detect(an_img_rdd_element):
    x = an_img_rdd_element[0]
    img = an_img_rdd_element[1]
    img_fname = x.split("/")[-1]
    file_bytes = np.asarray(bytearray(img), dtype=np.uint8)
    im = cv2.imdecode(file_bytes,1)
    #im = cv2.imread(img)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(SparkFiles.get("haarcascade_frontalface_default.xml"))
    faces = faceCascade.detectMultiScale(im)
    print faces
    print "Number of faces found in-> ", img_fname, " are ", len(faces)
    for (x,y,w,h) in faces:
        cv2.rectangle(im, (x,y), (x+w,y+h), (255,255,255), 3)
    rect_img_path = rect_img_dir + 'rect_' + img_fname
    cv2.imwrite(rect_img_path,im)
    return (img_fname,len(faces))

# Transformation of the image RDD
# Calling map function on the images RDD
num_face_rdd = images_RDD.map(face_detect)

#num_face_rdd.persist()

# Action on the transformed RDD
# Collecting the result of the map function
result = num_face_rdd.collect()
print result

# End time
end = timer()

# <codecell>

# Total time taken by this script
time_taken = (end-start)
print "Total time taken in detecting faces in "+ str(len(result)) + " images is "+ str(time_taken) +" seconds."

# Save the face detection result in the form of pickle
pickle.dump(result,open("./face_detection_result.p","wb"))

