import os

gen_img_path = "/home/default/KerasWorkspace/Keras-FCN/dataGenerated"
train_img_dir = "/home/default/.keras/datasets/weedspic/lettuce/image_aug"
train_lable_dir = "/home/default/.keras/datasets/weedspic/lettuce/label_aug"

x=0
y=0
k=0
while (x<9999):
 while (k<9999):
 	if(os.path.isfile("%s/img_test_%d_%d.png" % (gen_img_path,x,k)) and
   	   os.path.isfile("%s/label_test_%d_%d.png" % (gen_img_path,x,k))):

 		old_file = os.path.join("%s/img_test_%d_%d.png" % (gen_img_path,x,k))
 		new_file = os.path.join(train_img_dir, "%04d.png" % (y))
 		os.rename(old_file, new_file)
 	
 		old_file = os.path.join("%s/label_test_%d_%d.png" % (gen_img_path,x,k))
 		new_file = os.path.join(train_label_dir, "%04d.png" % (y))
 		os.rename(old_file, new_file)
                y+=1
 	k += 1
 x += 1
 k = 0


