import matplotlib.pyplot as plt
import numpy as np

f = open("exp1.txt","r")
list_of_strings = f.read().split("\n")
list_of_strings.pop(-1)
l1 = []

n = len(list_of_strings)
for i in range(n):
    s = list_of_strings[i].split()
    s1 = []
    for j in range(3):
        #print(s[j])
        num = eval(s[j])
        s1.append(num)

    l1.append(s1)

l1 = np.asarray(l1)
batch = l1[:,0]
acc1, acc2 = l1[:,1],l1[:,2]
plt.scatter(batch,acc1,label='batch accuracy')
plt.scatter(batch,acc2,label='batch balanced accuracy')
plt.xlabel("bacth bumber")
plt.ylabel("classification accuracy")
plt.legend()
plt.title("batch training accuracy during 1st epoch")
plt.savefig("batch_training.png")
plt.show()
        
