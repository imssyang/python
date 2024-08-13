# YOLO

"You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

[Redmon et al., 2016](https://arxiv.org/abs/1506.02640)
[Redmon and Farhadi, 2016](https://arxiv.org/abs/1612.08242).

# Pretrain

[Darknet is an open source neural network framework written in C and CUDA.](https://github.com/pjreddie/darknet)

```bash
wget http://pjreddie.com/media/files/yolov2.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg
```

# Dataset

[The PASCAL Visual Object Classes Challenge 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007)

```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/devkit_doc_07-Jun-2007.pdf
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```

## Anchor Boxes

Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the different classes. For this project, 5 anchor boxes were chosen  (to cover the 20 classes), and stored in the file './models/yolov2_anchors.txt'

* The dimension of the encoding tensor of the second to last dimension based on the anchor boxes is $(m, n_H,n_W,anchors,classes)$.
* The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).

## Encoding

What this encoding represents.

 <img width="824" alt="architecture" src="https://user-images.githubusercontent.com/72977734/204974620-b495b28a-ab31-4514-a617-6a579941d517.png">

 <caption><center> <u><b> Figure 2 </u></b>: Encoding architecture for YOLO<br> </center></caption>

 Since you're using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.

For simplicity, you'll flatten the last two dimensions of the shape (19, 19, 5, 85) encoding, so the output of the Deep CNN is (19, 19, 425).

<img width="791" alt="flatten" src="https://user-images.githubusercontent.com/72977734/204974727-23136d19-7fb2-4183-9f1d-a0e9ebcd27b7.png">

<caption><center> <u><b> Figure 3 </u></b>: Flattening the last two last dimensions<br> </center></caption>


## Class score

Now, for each box (of each cell) you'll compute the following element-wise product and extract a probability that the box contains a certain class.  
The class score is $score_{c,i} = p_{c} \times c_{i}$: the probability that there is an object $p_{c}$ times the probability that the object is a certain class $c_{i}$.

<img width="825" alt="probability_extraction" src="https://user-images.githubusercontent.com/72977734/204974954-523e5b52-bbf0-497f-85de-9cf0d9f2b08b.png">

<caption><center> <u><b>Figure 4</u></b>: Find the class detected by each box<br> </center></caption>


## Visualizing classes
Here's one way to visualize what YOLO is predicting on an image:

- For each of the 19x19 grid cells, find the maximum of the probability scores (taking a max across the 80 classes, one maximum for each of the 5 anchor boxes).
- Color that grid cell according to what object that grid cell considers the most likely.

Doing this results in this picture: 

<img width="691" alt="proba_map" src="https://user-images.githubusercontent.com/72977734/204975079-140da382-addd-4ec2-966a-4b5a9ea7e0ce.png">

<caption><center> <u><b>Figure 5</u></b>: Each one of the 19x19 grid cells is colored according to which class has the largest predicted probability in that cell.<br> </center></caption>


## Visualizing bounding boxes
Another way to visualize YOLO's output is to plot the bounding boxes that it outputs. Doing that results in a visualization like this:

<img width="452" alt="anchor_map" src="https://user-images.githubusercontent.com/72977734/204975239-fadcf13d-58fb-48e8-a3f7-204582884a36.png">

<caption><center> <u><b>Figure 6</u></b>: Each cell gives you 5 boxes. In total, the model predicts: 19x19x5 = 1805 boxes just by looking once at the image (one forward pass through the network)! Different colors denote different classes. <br> </center></caption>

## Non-Max suppression
In the figure above, the only boxes plotted are ones for which the model had assigned a high probability, but this is still too many boxes. You'd like to reduce the algorithm's output to a much smaller number of detected objects.  

To do so, you'll use **non-max suppression**. Specifically, you'll carry out these steps: 
- Get rid of boxes with a low score. Meaning, the box is not very confident about detecting a class, either due to the low probability of any object, or low probability of this particular class.
- Select only one box when several boxes overlap with each other and detect the same object.

Even after filtering by thresholding over the class scores, you still end up with a lot of overlapping boxes. A second filter for selecting the right boxes is called non-maximum suppression (NMS). 

<img width="779" alt="non-max-suppression" src="https://user-images.githubusercontent.com/72977734/204975359-cef80581-1092-48c0-87f2-6df0c4bf1abf.png">

<caption><center> <u> <b>Figure 7</b> </u>: In this example, the model has predicted 3 cars, but it's actually 3 predictions of the same car. Running non-max suppression (NMS) will select only the most accurate (highest probability) of the 3 boxes. <br> </center></caption>

Non-max suppression uses the very important function called **"Intersection over Union"**, or IoU.
<img src="nb_images/iou.png" style="width:500px;height:400;">

<caption><center> <u> <b>Figure 8</b> </u>: Definition of "Intersection over Union". <br> </center></caption>



