# deep-learning-capstone
Deep Learning Capstone Project for Udacity's Machine Learning Nanodegree

# How to run it
You need tensorflow installed in your system to run it. TensorFlow only supports Linux and Mac right now.

I used Python2 to write this project. You need to create a ./data dir at your working dir

Open your terminal
```
python dp.py
```
This will run the default routine.

You can also specify the exact parameters to run. Create a new pytohn file
```
import dp
net = dp.Net(
	num_hidden=64,
	batch_size=64,
	patch_size=7,
	conv1_depth=16,
	conv2_depth=16,
	pooling_stride=2,
	drop_out_rate=0.9,
	num_steps=5001,
	optimizer='momentum',
	base_learning_rate=0.0013,
	decay_rate=0.99,
	train_csv='record/train.csv', test_csv='record/test.csv',
	model_name='model.ckpt'
)
net.train()
net.test()
```
You can of course just open dp.py and modify code directly in the secion
```
if __name__ == '__main__':
    # Your code here
```
After you call net.train(), a model with your specified name will be created in ./data dir

Then you can call net.test() to resotre the model without calling train() again.

# Note
git ignores .mat files. Please download data sets from http://ufldl.stanford.edu/housenumbers/

Then create data/ dir and put them in this dir

You need train_32x32.mat and test_32x32.mat

Report is in report/report.pdf

record/ contains all the result data

chart/ contains all the charts


