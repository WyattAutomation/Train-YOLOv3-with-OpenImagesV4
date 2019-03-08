# Training YOLOv3 to detect specific objects using Google's OpenImagesV4

This is a detailed tutorial on how to download a specific object's photos with annotations, from Google's Open ImagesV4 Dataset, and how to fully and correctly prepare that data to train PJReddie's YOLOv3.  This how I trained this model to detect "Human head", as seen in the GIF below:

![Alt Text](https://github.com/WyattAutomation/Train-YOLOv3-with-OpenImagesV4/blob/master/yoloDemo.gif?raw=true)

## Getting Started

### This README explains how to use the CSVheadstoTXT.py script to train YOLOv3 to detect a single object/class.  I have now added a new script (MULTI-CLASSjpg2txt.py) that can prep data for multiple classes.  I haven't updated the below documentation for manually prepping the config files, but the new script is commented to explain how to use it, and adding extra classes should be self-explanatory per the below doc.  If it becomes apparent that is not the case, put in a ticket and I'll append it as needed. -GW

Make sure you have PJReddie's YoloV3 installed, compiled with CUDA and OpenCV, and working with your webcam for live video.  The version of YoloV3 I am using installed is from here:
```
https://pjreddie.com/darknet/yolo/
```

You will need Pandas for Python3.6 installed as well.  

I'm using CUDA Toolkit 10.0 with the 410 version of the NVidia Driver, on Ubuntu18.04, and use Python 3.6.  My GPU is an NVidia gtx 1060 (6GB model).  I won't go into details too much with getting YOLOv3 working, as that is not the intent of this document.


## Downloading Photos Containing only Specific Classes of Objects from the Open Images Dataset:

If you don't want to download the entire Google Open Images Dataset, and just need photos and annotations containing one or a few objects that you can seach for on their "Explore" page, I will explain how to do this first. (The 'explore' page I'm talking about is found here: https://storage.googleapis.com/openimages/web/visualizer/index.html?)

### OIDv4 Tookit
There are several ways to do this but I will explain the easiest that I've found so far, which is by using the OIDv4toolkit, This is a downloader tool specifically made for this purpose (https://github.com/EscVM/OIDv4_ToolKit).  The documentation on their GitHub isn't entirely accurate, so please follow my instructions here!

Clone the repo to whatever directory you want.  Desktop is fine:
```
git clone https://github.com/EscVM/OIDv4_ToolKit.git
```
Check the "requirements.txt" file for the dependencies.  I would reccomend against running "pip3 install -r requirements.txt"; if you are running a custom compiled version of OpenCV like I am, and are not using a Python virtual env, it can screw things up pretty badly, so just open requirements.txt and install each manually, skipping things you know you already have.  It's all very common stuff and you probably already have it installed anyway.

Once you have the requirements installed, it's time to run the downloader (main.py from the root directory of the tool).

Note that the correct command below uses "downloader" and not "download" like the documentation on the OIDv4 GitHub erroneously states.  Also note, if there is a space in the class name like "Human head" type it like "Human_head" below:
```
python3 main.py downloader --classes Human_head --type_csv all
```
If it asks:
```
[ERROR] Missing the train-annotations-bbox.csv file.
[DOWNLOAD] Do you want to download the missing file? [Y/n] 
```
Answer Y to all 3 of these, whenever you encounter them (it should prompt just before it downloads the training, test, and validation CSV files).

### Annotation CSVs Explained
These files it asks to download are the CSV files (that I will refer to as 'annotation CSVs') that contain the information about where the objects are (the bounding boxes of the objects) in the corresponding photos you're downloading.  The important information on each row of this CSV includes:

-ImageID - the name of the actual .jpg file in the dataset, without '.jpg' on the end

-LabelName - a 'Label Name' which is just a label that looks like '/m/04hgtk' and represents the type of object that's in the photo.  You can open 'class-descriptions-boxable.csv' in a text editor and search for the object you want and find it's LabelName

-XMin - lowest value of X for the bounding box (box that conatins object) that the object is in

-XMax - highest value of X for the bounding box that the object is in

-Ymin - lowest value of Y for the bounding box that the object is in

-Ymax - highest value of Y for the bounding box that the object is in

The other information in these CSVs, we will not use here.

It's important to note that these CSV files contain this information for ALL of the photos in the Open Images Dataset, but don't worry as I've included tools here to get information only about the photos we are downloading, and to refine that info even further to fully prepare the data.

If you start downloading and see a bunch of aws or other errors flying accross the terminal that move too fast to read, it isn't working correctly.  It should just look like download-progress bars in the terminal like this:
![Alt Text](https://github.com/WyattAutomation/Train-YOLOv3-with-OpenImagesV4/blob/master/dlimage.png?raw=true)

If you downloaded Human_head, and you cloned the OIDv4Toolkit/ folder to your desktop you should have a directory structure like this:
```
OIDv4_ToolKit
│   main.py
│
└───OID
    │
    └───csv_folder
    |    │   class-descriptions-boxable.csv
    |    │   validation-annotations-bbox.csv
    |	 |	 test-annotations-bbox.csv	
    |	 |	 train-annotations-bbox.csv
    └───Dataset
        |
        └─── test
             |
             └───Human head
                   |
                   |0fdea8a716155a8e.jpg
                   |2fe4f21e409f0a56.jpg
                   |...
                   └───Labels
                          |
                          |0fdea8a716155a8e.txt
                          |2fe4f21e409f0a56.txt
                          |...
              
        |
        └─── train
             |
             └───Human head
                   |
                   |0fdea8a716155a8e.jpg
                   |2fe4f21e409f0a56.jpg
                   |...
                   └───Labels
                          |
                          |0fdea8a716155a8e.txt
                          |2fe4f21e409f0a56.txt
                          |...
              
        |
        └─── validation
             |
             └───Human head
                   |
                   |0fdea8a716155a8e.jpg
                   |2fe4f21e409f0a56.jpg
                   |...
                   └───Labels
                          |
                          |0fdea8a716155a8e.txt
                          |2fe4f21e409f0a56.txt
                          |...
              
```
Note that it will create a directory called "labels" that we do not need, you can ignore the txt files and that directory as they are not used.  There is an option to not create them as well, but it doesn't hurt anything if they do get created.  Also note that the names in the files above are just arbitrary names here for demonstration purposes.  

We will also not be using anything but the "training" data in this guide; future releases will include how to improve the creation of trained weights using the others, but for now the only CSVs and images we will be using are the "Training" datasets.


## Create the .data, the .cfg, and the .names files:

### .data File
I wanted to train mine to reconize one object, 'Human head', so I created a text file in the /darknet/cfg/ directory of yolo and named it 'head.data' containing the following:

```
classes= 1
train  = /home/sbubby/train.txt
valid  = /home/sbubby/test.txt
names = data/head.names
backup = /home/sbubby/backup
```
**Note that 'sbubby' above is my username; change sbubby to your username for wherever the directories of the files you use here are going to be located on your machine, for example: /home/YourUserNameGoesHere/train.txt

-classes is '1' as I'm only training mine to detect 'Human head'.  This is the only thing here that's a parameter, the rest are file locations.

-train.txt is a single text file that lists the full directory of where each photo is that you downloaded; we'll go
through the process of creating that later. 

-test.txt; same as above and this is created along with the train.txt file in a later process

-"backup" is just a folder where you want the trained weights files to be output to while it trains.  Make sure this folder exists, and that you don't put an extra "/" after /home/sbubby/backup like a lot of other tutorials say to.

**MAKE SURE that before you start training, all of the files that this .data file points to, end up being where where it says they need to be!!  If you have issues during training where something can't be found, make sure both the paths here are correct and the files you create in this tutorial are located at those paths!  

**You can put the test.txt, train.txt, and backup folder wherever on your machine, as long as this file points to their locations and can access them  you're good (putting them in the directory of darknet is fine or /home/username/ like I did above is fine.  DON'T stick it somewhere dumb like /etc, a hidden folder, or somewhere that requires admin priviledges etc to access!!!

### .names File

-head.names: a text file that you name "whateveryouwant.names", that contains a list of the names of the objects
that you are going to train yolo to detect.  Since we are only training it to detect "Human head", that file should
just contain:
```
Human head
```
at the top, with a space after it (not indented or anything too, just at the top left).  If you were doing this for
2 objects like Human head and Human hand, it'd be like:
```
Human head
Human hand
```

-If I were using the .data file from before, I would stick this file in the ../darknet/data folder of your yolo installation.


### .cfg File
Create the cfg by making a copy of the existing yolov3.cfg file in the /darknet/cfg directory and naming it "whateveryouwant.cfg", I named mine "head.cfg", and put it in the same /cfg directory.

Edit that new cfg as follows:
-Line 3: set batch=16, this means we will be using 16 images for every training step

-Line 4: set subdivisions=16, the batch will be divided by 16 to decrease GPU VRAM requirements.

-Line 603: set filters=(numberOfclasses + 5)*3 in our case (for one object) filters=18

-Line 610: set classes=1, the number of categories we want to detect

-Line 689: set filters=(numberOfclasses + 5)*3 in our case filters=18

-Line 696: set classes=1, the number of categories we want to detect

-Line 776: set filters=(numberOfclasses + 5)*3 in our case filters=18

-Line 783: set classes=1, the number of categories we want to detect

Save that after making the changes, and put it in your ../darknet/cfg folder of your yolo installation if it isn't there already.


## Creating the proper txt files for each image:

### Edit CSVheadstoTXT.py: 
-Change line 5 of this included python script in my repo to point to the path of the training annotation CSV you donwloaded earlier. So for the 'training' images, if your /OIDv4_ToolKit folder was on my (sbubby's) desktop then:
```
f=pd.read_csv("/home/sbubby/Desktop/OIDv4_ToolKit/OID/csv_folder/train-annotations-bbox.csv")
```
Make sure that you change "sbubby" here to your username, and/or change the directory to wherever your train-annotations-bbox.csv file was downloaded (this is the default dl location of the csv files if you cloned OIDv4 to your Desktop).

**Again, this script only preps training data for one class of object at a time for now.  I will be updating this in a day or two to make it prep the data for multiple classes, so please bare with me!

## For prepping data for a single class:

-Change line 7 to use the objects' LabelName.  It's set to use the LabelName for "Human head" in the script, so change this if your object is different.  It will look like '/m/04hgtk' and you can find these labels in the CSV that's in the  OIDv4_ToolKit/OID/csv_folder/class-descriptions-boxable.csvclass-descriptions-boxable.csv from the toolkit.

-Change line 36 to point to the directory to dump the txt's that are generated for each image.  So for example I created a directory called "yolotxt" and set it to:
```
/home/sbubby/Desktop/OIDv4_ToolKit/OID/Dataset/train/yolotxt/
```
-Copy CSVheadstoTXT.py to the same directory as the .jpg files you downloaded.  For me in this example I copied like this: 
```
cp CSVheadstoTXT.py /home/sbubby/Desktop/oiheadpics/OIDv4_ToolKit/OID/Dataset/train/Human \head
```
-Obviously change the above to the right directory on your machine, then cd to that directory above and run it:
```
python3 CSVheadstoTXT.py
```
-This should generate all of the .txt files mentioned above.  For my example here, it dumped them all to /home/sbubby/Desktop/oiheadpics/OIDv4_ToolKit/OID/Dataset/test/yolotxt/  

-Once it's done, copy all those .txt files to the same directory as the .jpg files, so like: 
```
cp -r /home/sbubby/Desktop/oiheadpics/OIDv4_ToolKit/OID/Dataset/test/yolotxt/. /home/sbubby/Desktop/oiheadpics/OIDv4_ToolKit/OID/Dataset/test/Human\ head/
```
as the .txt's are in /yolotxt and the .jpg's are in /Human\ head ("Human\ head" is written like this as there will be a sapce in the name)


## Details of Info in .txt Files

-Each .txt file should have the same name as the image it corresponds to, except with ".txt" at the end, so for "00a0fd8177a1db74.jpg" you should get "00a0fd8177a1db74.txt" from this, in the /yolotxt directory above.

-Each .txt file should contain a new row for each object in the picture that it was made for, so for "00a0fd8177a1db74.jpg" which is a photo that contains two "Human head" objects, it would have a corresponding file in the /yolotxt folder called "00a0fd8177a1db74.txt" that contains:
```
0 0.359375 0.4145835 0.07812600000000003 0.141667
0 0.554687 0.3958335 0.09375 0.145833
```
-The first column is 0, which is the row # (starting at 0) that the object's name is found on in your ".name" file that was made above.  **If you're only training to detect ONE object, this will be 0 in every text file in every row, as there will only be one object in the head.names file.**
-the second column is the middle X point of the object's bounding box, found by adding XMin and XMax and dividing their sum by 2
-The third column is the middle Y point of the object's bounding box, found by adding YMin and YMax and dividing their sum by 2
-The fourth column is the width, found by subtracting XMin from XMax
-The fifth column is the height, found by subtracting YMin from YMAx

The script should take care of all of this for you, as long as the directories are set up correctly. I'm just explaining it here for refference.  Feel free to look at the script and see how pandas is used, and what basic arithmetic is used to proccess the data; it's quite simple.


**IMPORTANT TO NOTE: *MANY* other tutorials on training Yolov3 on your own datasets include x and y pixel values that have to be converted to 0-1 relative values.  So if you had a photo width of 1280, the "absolute x" value for the mid point would be 640, and the "realtive x" for the mid point would be .5 .  Open Images already uses values between 0 and 1 in their annotations, 0 being the lowest possible value in the whole photo for x or y, and 1 being the highest possible value.  These are called "relative" x and y, as they represent the grid locations of things in the photo by values of 0-1 along the x and y axies relative to a percetage of the max x or y, rather than pixel value coordinates. This is due to the fact that the photos in this huge dataset are of many different sizes.  Anyway, if you come accross tutorials claiming that you need to calculate the "relative" values, like what 714/1280 is for the mid points, DON'T DO THAT HERE!!  The data is already in the correct format!


## Create the single txt for your '.data' file, containing all the images paths:

-In process.py, change line 5 to the directory where the .jpg files for the training dataset for the object are.  For me, this looked like:
```
current_dir = '/home/sbubby/Desktop/OIDv4_ToolKit/train/Human head'
```
Change it to wherever that directory is on your machine, and run:
```
process.py
```
-This should dump a train.txt and test.txt that you'll need the .data file to point to.  It should just dump them both wherever proccess.py was ran from.

-Copy those two files to where your .data file earlier in this Tutorial is configured to look for them 


## Download Weights for Convolutinal Layers

-Just download this from the following link, and copy it into the root directory of yolo (/darknet):
https://pjreddie.com/media/files/darknet53.conv.74

## Train It!

Once all of this is complete command to train:
```
./darknet detector train cfg/head.data cfg/head.cfg darknet53.conv.74
```

**If you get an error about running out of memory while it trains, you can tweak "batch=16" and "subdivisions=16", so batch is lower and subdivisions is higher.  This will make training take longer, but in this example I let it run all night (about 6 hours) with both set to 16. I may not have needed to wait that long, it may have been fine after 3: I'm not sure.  I'll reccomend with training in this guide that you test out training for about an hour, and if it hasn't stopped itself after that long (when it's working like it should, it will just keep going until you stop it), you should be good to leave it.  

**Note: If you used the above .data file example for Human head, it'd dump a .backup file into the directory that is specified by the last line in that head.data example above.  The .backup file is the pretrained weights when you stop the training, the _100.weights _200.weights files are the weights after specific intervals.  I've discovered that PJReddies appears to either not generate a new .weights after a very long time, or that 900 is the last before it stops making backups of the weights.  Either way, I normally just just the .backup file as that's the farthest it got to when you stop the training.  There are reasons to avoid over-training but I won't get into that here, just use the .backup file and if it's problematic try the others.


Start it before you go to bed, before you go to work, whatever you want just let it do it's thing and I'd suggest not using your computer while it runs..  

..now's your chance, go live your LIFE, at least for several hours!..  You've been behind a screen for a minute or two if you're new to this and made it this far!

To run the demo of your weights in live video after you've stopped the training, follow the instructions for compiling yolov3 with CUDA and OpenCV, then run:
```
./darknet detector demo cfg/head.data cfg/head.cfg backup/head.backup
```
Should show you a window like the one at the beginning, with a feed of your webcam and bounding boxes drawn around all detected "Human heads" or whatever you trained it to detect!!  

Have fun!

-Gene
