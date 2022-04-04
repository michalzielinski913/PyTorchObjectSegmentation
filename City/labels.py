#!/usr/bin/python
#
# Converts the polygonal annotations of the Cityscapes dataset
# to images, where pixel values encode the ground truth classes and the
# individual instance of that classes.
#
# The Cityscapes downloads already include such images
#   a) *color.png             : the class is encoded by its color
#   b) *labelIds.png          : the class is encoded by its ID
#   c) *instanceIds.png       : the class and the instance are encoded by an instance ID
#
# With this tool, you can generate option
#   d) *instanceTrainIds.png  : the class and the instance are encoded by an instance training ID
# This encoding might come handy for training purposes. You can use
# the file labes.py to define the training IDs that suit your needs.
# Note however, that once you submit or evaluate results, the regular
# IDs are needed.
#
# Please refer to 'json2instanceImg.py' for an explanation of instance IDs.
#
# Uses the converter tool in 'json2instanceImg.py'
# Uses the mapping defined in 'labels.py'
#

# python imports
from __future__ import print_function, absolute_import, division
import os, glob, sys

# cityscapes imports
def createInstanceImage(annotation, encoding):
    # the size of the image
    size = ( annotation.imgWidth , annotation.imgHeight )

    # the background
    if encoding == "ids":
        backgroundId = name2label['unlabeled'].id
    elif encoding == "trainIds":
        backgroundId = name2label['unlabeled'].trainId
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # this is the image that we want to create
    instanceImg = Image.new("I", size, backgroundId)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw( instanceImg )

    # a dict where we keep track of the number of instances that
    # we already saw of each class
    nbInstances = {}
    for labelTuple in labels:
        if labelTuple.hasInstances:
            nbInstances[labelTuple.name] = 0

    # loop over all objects
    for obj in annotation.objects:
        label   = obj.label
        polygon = obj.polygon

        # If the object is deleted, skip it
        if obj.deleted:
            continue

        # if the label is not known, but ends with a 'group' (e.g. cargroup)
        # try to remove the s and see if that works
        # also we know that this polygon describes a group
        isGroup = False
        if ( not label in name2label ) and label.endswith('group'):
            label = label[:-len('group')]
            isGroup = True

        if not label in name2label:
            printError( "Label '{}' not known.".format(label) )

        # the label tuple
        labelTuple = name2label[label]

        # get the class ID
        if encoding == "ids":
            id = labelTuple.id
        elif encoding == "trainIds":
            id = labelTuple.trainId

        # if this label distinguishs between individual instances,
        # make the id a instance ID
        if labelTuple.hasInstances and not isGroup and id != 255:
            id = id * 1000 + nbInstances[label]
            nbInstances[label] += 1

        # If the ID is negative that polygon should not be drawn
        if id < 0:
            continue

        try:
            drawer.polygon( polygon, fill=id )
        except:
            print("Failed to draw polygon with label {} and id {}: {}".format(label,id,polygon))
            raise

    return instanceImg

def json2instanceImg(inJson,outImg,encoding="ids"):
    annotation = Annotation()
    annotation.fromJsonFile(inJson)
    instanceImg = createInstanceImage( annotation , encoding )
    instanceImg.save( outImg )

# The main method
def main():
    # Where to look for Cityscapes

    cityscapesPath= 'F:\CityDataset'
    # how to search for all ground truth
    searchFine   = os.path.join( cityscapesPath , "gtFine"   , "*" , "*" , "*_gt*_polygons.json" )
    searchCoarse = os.path.join( cityscapesPath , "gtCoarse" , "*" , "*" , "*_gt*_polygons.json" )

    # search files
    filesFine = glob.glob( searchFine )
    filesFine.sort()
    filesCoarse = glob.glob( searchCoarse )
    filesCoarse.sort()

    # concatenate fine and coarse
    files = filesFine + filesCoarse
    # files = filesFine # use this line if fine is enough for now.

    # quit if we did not find anything
    if not files:
        print( "Did not find any files. Please consult the README." )

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files:
        # create the output filename
        dst = f.replace( "_polygons.json" , "_instanceTrainIds.png" )

        # do the conversion
        try:
            json2instanceImg( f , dst , "trainIds" )
        except:
            print("Failed to convert: {}".format(f))
            raise

        # status
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()


# call the main
if __name__ == "__main__":
    main()