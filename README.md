# denseFood: Food Recognition Based on Densely Connected Convolutional Networks
<img src="img-1.jpg"/>
<h1> Introduction </h1>
<p>
In the latest years, there is dramatic consideration on using technology in many fields especially when we use the artificial intelligence to make our life easy. Due to the increase using of computer vision technology in many domains like surveillance cameras, healthcare, etc.. Food recognition is one of these important fields and deserves more research efforts because of its practical importance and scientific challenges
</p>
<p>
Recently, Convolutional Neural Network (CNN) is used in the context of food recognition. Food recognition methods uses CNN models to extract food image features, compute the similarity of food image features and use classification techniques to train the classifier to accomplish food recognition.
</p>
<h1>Proposed Method</h1>
<img src="dense_connectivity.JPG"/>
<p>
The proposed model presented called DenseFood, which is based on DenseNet architecture
</p>
<h1>Experiment</h1>
<p>
Let's to divide the dataset into two sub dataset for training and testing by run this script
</p>
<pre>$ python build_datse_food.py --data_dir  --output_dir</pre>
<p>--data_dir refers to the source dataset folder</p>
<p>--output_dir refers to the output folder to split dataset into</p>

<h3> Training  </h3>
<p>
We will train our model by run the script as below. we need to choose which model we will use to train by specievied the --model_dir that contain the params.json
</p>
<pre>
$ python train.py --data_dir food172 --model_dir experiments/densefood
</pre>

<p> Food Evaluation.ipynb file contain a code for data visualization</p>