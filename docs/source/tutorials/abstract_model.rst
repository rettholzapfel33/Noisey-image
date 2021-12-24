Integrating New Network
==========================

In this tutorial, we are going to look at how to take an existing network and integrate it with the GUI interface.

General Overview
-----------------
In a general sense, all applicable neural networks (CNN-based Object Detectors, Semantic Segmentation, and etc.) can have their actions boiled down to the following:

* Initialization
* Inference (batchsize variable)
* Render (draw) output
* Deinitialization (optional)

The `Model` abstract class given in the models.py provide a skeleton that can make integration very easy to do.
Create a class (perferably in the same models.py file) and inherit the Model class. From then, you will have to reimplement the following functions:

* `initialize(\*kwargs)`
* `run(input)`
* `draw(prediction, img)`
* `deinitialize()`

After creating the inherited class, create an initialized object of the class in the `_registry` variable.

Example
---------------------------

For this example, let's say we are (re)implementing and reintegrating the `YOLOv3` (You Look Only Once) network.
This network was originally created with the Darknet Neural Network framework, which is created using C++.

To integrate the C++ functionalities into PyQt5 is quite difficult, as there is an issue with conflicting CDLL loading from both Qt5 and Darknet.
For this reason, we are using this PyTorch port of YOLOv3: https://github.com/eriklindernoren/PyTorch-YOLOv3

This particular implementation's initialization sequence is the following:
* Converting configuration (cfg) file from text to a sequence of PyTorch layers
* Load in binary Darknet weight (weights) file, convert into same structure as PyTorch network
* Load in converted weights into PyTorch network

So, to start the process of integration, let's create our inherited class:

.. code-block:: python
    :caption: Creating Inherited Class and Implementing the Initialization Function

    class YOLOv3(Model):
        def __init__(self, network_config):
            '''
            Assume network_config is a tuple:
            (CFG: str, WEIGHTS: str, CLASSES: str)
            '''
            self.CFG = network_config[0]
            self.WEIGHTS = network_config[1]
            self.CLASSES = network_config[2]
        
        def run(self, input):
            return input # TODO later in tutorial

        def initialize(self, *kwargs):
            # load-model is a function from the PyTorch YOLOv3 implementation
            self.yolo = load_model(self.CFG, self.WEIGHTS)
            # load_classes is the same as load_model
            self.classes = load_classes(self.CLASSES)
            return 0
        
        def deinitialize(self):
            return 0 # Optional TODO

        def draw(self, pred, img):
            return 0 # TODO later in tutorial

To get the network detecting images, we can utilize the following function to transform and run the image through the network:

`detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5)`

The `run` function would turn to:

.. code-block:: python
    :caption: Updating Inherited Class with Detection Method

        class YOLOv3(Model):
        def __init__(self, network_config):
            ...
        
        def run(self, input):
            pred = detect.detect_image(self.yolo, input)
            return pred
        ...

Finally, we would want to implement the draw and deinitialize functions. For the draw function, since it is an object detector, we should loop through the detections, take the coordinates, and draw the rectangles.
Deinitialization is optional and only really applies for certain cases (long-sequence, multi-network pipelines). In this case, it does not need to be implemented.

.. code-block:: python
    :caption: Updating Inherited Class with Draw Method

        class YOLOv3(Model):
        def __init__(self, network_config):
            ...
        
        def draw(self, pred, img):
            np_img = np.copy(img)
            for bbox_cls in pred:
                bbox, cls = bbox_cls
                x1, y1, x2, y2 = list(map(int, bbox))
                np_img = cv2.rectangle(np_img, (x1,y1), (x2,y2), (0,0,255), 2)

            return np_img

After the YOLOv3 Model class has been created, you can create an object of that network (note: object creation is **not** the same thing as initializing a network)
within the `_registry` dictionary:

.. code-block:: python
    :caption: Insertion of YOLOv3 class object
        
        _registry = {
            'yolov3': YOLOv3(
                os.path.join('obj_detector/cfg', 'coco.names'),
                os.path.join('obj_detector/cfg', 'yolov3.cfg'),
                os.path.join('obj_detector/weights','yolov3.weights')
            )
        }