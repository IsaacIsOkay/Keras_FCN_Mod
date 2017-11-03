f = open('train.txt', 'w')


numberOfImages = 10093
for x in range(0, numberOfImages):
	f.write("%04d\n" % (x))
f.close()

