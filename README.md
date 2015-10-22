Matcaffe vs. Pycaffe
#####################
This project is to show a case whether Matcaffe and Pycaffe would give the same prediction if we provide a same input. In short, my conclusion is that Matcaffe and Pycaffe would different predictions and different confidences.

## Generate input
I use an example image 0006.jpg in this folder and use function prepare_image Matcaffe [demo](https://github.com/BVLC/caffe/blob/master/matlab/demo/classification_demo.m) to sample ten patches (do mean substract, sampling 4 corners and center, and flipping). Model used to do classification is [CaffeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet).

![Image](0006.jpg)

## Discrepency in the results
I highlight differences in Matcaffe and Pycaffe results as the following. 

Matcaffe results
```
Predicted class of center cropping is 425.000000.
Confidence of prediction is 0.359984.
Final prediction of ten croppings is 425.000000
Confidence of prediction is 0.406270.
```

Pycaffe resutls
```
Predicted class of center cropping is #424.
Confidence of prediction is #0.111404724419.
Final prediction of ten croppings is #702.
Confidence of prediction is #0.103248856962.

Predicted class of pycaffe preprocess method is #424.
Confidence of prediction is #0.578046441078.
```

## Possible explanations or reasons.
Currently, I do not have knowledge to explain it. Hope Caffe developers would help to answer it.