import os

x=0
y=0
k=0
while (x<9999):
 while (k<9999):
 	if(os.path.isfile("/home/default/KerasWorkspace/Keras-FCN/dataGenerated/img_test_%d_%d.png" % (x,k)) and
   	   os.path.isfile("/home/default/KerasWorkspace/Keras-FCN/dataGenerated/label_test_%d_%d.png" % (x,k))):

 		old_file = os.path.join("/home/default/KerasWorkspace/Keras-FCN/dataGenerated/img_test_%d_%d.png" % (x,k))
 		new_file = os.path.join("/home/default/.keras/datasets/weedspic/lettuce/image_aug", "%04d.png" % (y))
 		os.rename(old_file, new_file)
 	
 		old_file = os.path.join("/home/default/KerasWorkspace/Keras-FCN/dataGenerated/label_test_%d_%d.png" % (x,k))
 		new_file = os.path.join("/home/default/.keras/datasets/weedspic/lettuce/label_aug", "%04d.png" % (y))
 		os.rename(old_file, new_file)
                y+=1
 	k += 1
 x += 1
 k = 0


