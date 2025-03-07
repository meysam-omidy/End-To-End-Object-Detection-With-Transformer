![2025-03-07 13_12_24-End-to-End Object Detection with Transformers 2020 pdf and 1 more page - Persona](https://github.com/user-attachments/assets/b753fa47-4bac-4928-8238-b0d20a9098f0)

This repository contains the implementation of the DETR (DEtection TRansformer) model, as described in the paper "End-to-End Object Detection with Transformers" by Nicolas Carion et al. from Facebook AI. ([link to the paper](https://arxiv.org/abs/2005.12872))

DETR is a novel approach to object detection that views it as a direct set prediction problem. The model uses a transformer encoder-decoder architecture to predict a set of bounding boxes and class labels in parallel, without the need for many hand-designed components like non-maximum suppression or anchor generation.
The key features of DETR are:
- A set-based global loss that forces unique predictions via bipartite matching
- A transformer encoder-decoder architecture that reasons about the relations of the objects and the global image context

## Dataset

This implementation was trained and evaluated on the COCO 2017 object detection dataset. COCO is a widely used benchmark for object detection, containing over 118,000 training images and 5,000 validation images, with a total of 80 object categories. ([link to the dataset](https://cocodataset.org/#download)

Files hierarchy after the extraction of dataset, should be like this:
```
.detr.py
.transformer.py
.coco_train.ipynb
\annotations
    .person_keypoints_train2017.json
    .instances_train2017.json
\images
    \train2017
        .000000000000.jpg
        .000000000001.jpg
```

## Requirements

To run the code, ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- Numpy
- Pycocotools (for loading the dataset)
- Cuda (preferred to train the model faster)

## Citation

If you use this code or find the DETR model useful, please consider citing the original paper:

```
@article{carion2020end,
  title={End-to-End Object Detection with Transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  journal={arXiv preprint arXiv:2005.12872},
  year={2020}
}
