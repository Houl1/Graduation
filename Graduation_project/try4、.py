from sklearn.datasets import load_svmlight_file
import pdb
def make_bread():
    pdb.set_trace()
data = load_svmlight_file("./lib/BIG15_vgg16.txt")
print(data[0].A[0])
print(len(data[1]))



