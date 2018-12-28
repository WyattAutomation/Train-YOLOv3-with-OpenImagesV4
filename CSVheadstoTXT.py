import numpy as np
import pandas as pd
import os

f=pd.read_csv("/home/sbubby/Desktop/OIDv4_ToolKit/OID/csv_folder/train-annotations-bbox.csv")
#for a single class, use this one
u = f.loc[f['LabelName'] == '/m/04hgtk']
keep_col = ['ImageID','XMin','XMax','YMin','YMax']

#For multiple classes use the below, adding as many new LabelNames as needed
#numClasses = ['/m/04hgtk','/m/0k65p']
#u = f.loc[df['LabelName'].isin(numClasses)]
#keep_col = ['LabelName', ImageID','XMin','XMax','YMin','YMax']

new_f = u[keep_col]
new_f['width'] = new_f['XMax'] - new_f['XMin']
new_f['height'] = new_f['YMax'] - new_f['YMin']
new_f['x'] = (new_f['XMax'] + new_f['XMin'])/2
new_f['y'] = (new_f['YMax'] + new_f['YMin'])/2
keep_col = ['ImageID','x','y','width','height']
new_f_2 = new_f[keep_col]

for root, dirs, files in os.walk("."):  
	for filename in files:

		if filename.endswith(".jpg"):
			fn = filename[:-4]
			nf = new_f_2.loc[new_f_2['ImageID'] == fn]
			#if only training one class
			nf['class_name'] = 0
			#If training multiple
			#nf['class_name'] = numClasses.index(nf['LabelName'])
			keep_col = ['class_name','x','y','width','height']
			new_nf = nf[keep_col]
			print(nf)
			imgpath = "/home/sbubby/Desktop/OIDv4_ToolKit/OID/Dataset/train/yolotxt/" + fn + ".txt"
			print(imgpath)

			new_nf.to_csv(imgpath, index=False, header=False, sep=' ')
			#pull the x,y,width,height data, for each row with the imageid, to a variable
			continue
		else:
			continue


#df = pd.DataFrame("train-annotations-bbox.csv")
#To select rows whose column value equals a scalar, some_value, use ==:
#f = pd.DataFrame(raw_data)
#f.head()
#f.loc[df['LabelName'] == '/m/04hgtk']

#new_f.loc[new_f['LabelName'] == '/m/04hgtk']
#new_f.to_csv("headstest.csv", index=False)