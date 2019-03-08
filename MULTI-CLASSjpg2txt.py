import pandas as pd
import os

f=pd.read_csv("/home/sbubby/Desktop/OID/csv_folder/train-annotations-bbox.csv")

#For multiple classes use the below, adding as many new LabelNames as needed
#this one is beer[0] cat[1] banana[2] in that order
numClasses = ['/m/01599','/m/01yrx','/m/09qck']
u = f.loc[f['LabelName'].isin(numClasses)]
keep_col = ['LabelName','ImageID','XMin','XMax','YMin','YMax']

new_f = u[keep_col]

new_f['ClassNumber'] = new_f['LabelName']

# adding a new column for Classnumber and setting the values based on LabelName
# so, for this, it's beer[0] cat[1] banana[2] in that order
new_f.loc[new_f['LabelName'] == '/m/01599', 'ClassNumber'] = 0
new_f.loc[new_f['LabelName'] == '/m/01yrx', 'ClassNumber'] = 1
new_f.loc[new_f['LabelName'] == '/m/09qck', 'ClassNumber'] = 2


new_f['width'] = new_f['XMax'] - new_f['XMin']
new_f['height'] = new_f['YMax'] - new_f['YMin']
new_f['x'] = (new_f['XMax'] + new_f['XMin'])/2
new_f['y'] = (new_f['YMax'] + new_f['YMin'])/2
keep_col = ['ClassNumber','ImageID','x','y','width','height']
new_f_2 = new_f[keep_col]

for root, dirs, files in os.walk("."):  
	for filename in files:

		if filename.endswith(".jpg"):
			fn = filename[:-4]
			nf = new_f_2.loc[new_f_2['ImageID'] == fn]
			keep_col = ['ClassNumber','x','y','width','height']
			new_nf = nf[keep_col]
			print(new_nf)
			imgpath = "/home/sbubby/Desktop/OIDLtool/OIDv4_ToolKit/OID/Dataset/train/yolotxtxt4all/" + fn + ".txt"
			print(imgpath)
			new_nf.to_csv(imgpath, index=False, header=False, sep=' ')
