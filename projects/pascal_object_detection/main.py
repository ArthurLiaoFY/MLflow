# %%
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms import v2

cudnn.benchmark = True

import matplotlib.pyplot as plt


def plot_image(source: str = "train", image_idx: str = "000012"):
    img = cv2.cvtColor(
        cv2.imread(
            filename=image_config[source][image_idx]["image_path"],
            flags=cv2.IMREAD_COLOR,
        ),
        cv2.COLOR_BGR2RGB,
    )
    for bounding_box, label in zip(
        *extract_xml_info(image_config[source][image_idx]["annotation_path"])
    ):
        cv2.rectangle(
            img=img,
            pt1=(bounding_box[0], bounding_box[1]),
            pt2=(bounding_box[2], bounding_box[3]),
            color=(0, 255, 0),
        )
        cv2.putText(
            img=img,
            text=label[0],
            org=(int((bounding_box[0] + bounding_box[2]) / 2), bounding_box[1]),
            fontFace=0,
            fontScale=0.7,
            color=(0, 255, 0),
        )

    plt.imshow(img)


def extract_xml_info(annotation_path: str, max_objects_per_image: int):
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        bounding_boxes = [
            [
                int(boxes.find("bndbox/xmin").text),
                int(boxes.find("bndbox/ymin").text),
                int(boxes.find("bndbox/xmax").text),
                int(boxes.find("bndbox/ymax").text),
            ]
            for boxes in root.iter("object")
        ]
        labels = [boxes.find("name").text for boxes in root.iter("object")]

        if len(labels) < max_objects_per_image:
            bounding_boxes = bounding_boxes + [
                [0, 0, 0, 0] for _ in range(max_objects_per_image - len(bounding_boxes))
            ]
            labels = labels + [
                "background" for _ in range(max_objects_per_image - len(labels))
            ]
        elif len(labels) > max_objects_per_image:
            bounding_boxes = bounding_boxes[:max_objects_per_image]
            labels = labels[:max_objects_per_image]

        return (
            tv_tensors.BoundingBoxes(
                bounding_boxes,
                format="XYXY",
                canvas_size=(
                    int(root.find("size/height").text),
                    int(root.find("size/width").text),
                ),
            ),
            labels,
        )

    except FileNotFoundError:
        return None, None


class PascalVOCImageDataset(Dataset):
    def __init__(
        self,
        image_config: dict,
        source: str = "train",
        do_transform: bool = False,
        is_inception_backbone: bool = False,
        max_objects_per_image: int = 20,
    ):
        self.idx_to_label = {
            idx: label
            for idx, label in enumerate(
                [
                    "aeroplane",
                    "bicycle",
                    "bird",
                    "boat",
                    "bottle",
                    "bus",
                    "car",
                    "cat",
                    "chair",
                    "cow",
                    "diningtable",
                    "dog",
                    "horse",
                    "motorbike",
                    "person",
                    "pottedplant",
                    "sheep",
                    "sofa",
                    "train",
                    "tvmonitor",
                    "background",
                ]
            )
        }
        self.label_to_idx = {v: k for k, v in self.idx_to_label.items()}
        self.image_config = image_config
        self.source = source
        self.do_transform = do_transform
        if is_inception_backbone:
            self.aim_height = 299
            self.aim_width = 299
        else:
            self.aim_height = 224
            self.aim_width = 224
        self.max_objects_per_image = max_objects_per_image

        # Important: In contrast to the other models
        # the inception_v3 expects tensors with a size of N x 3 x 299 x 299,
        # so ensure your images are sized accordingly.
        if self.do_transform:
            self.fit_transform = v2.Compose(
                [
                    v2.Resize(size=(self.aim_height, self.aim_width), antialias=True),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.fit_transform = v2.Compose(
                [
                    v2.Resize(size=(self.aim_height, self.aim_width), antialias=True),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def __len__(self):
        return len(self.image_config[self.source])

    def __getitem__(self, image_idx: str):
        # read image
        image = read_image(self.image_config[self.source][image_idx]["image_path"])
        # read annotation
        bounding_boxes, labels = extract_xml_info(
            annotation_path=self.image_config[self.source][image_idx][
                "annotation_path"
            ],
            max_objects_per_image=self.max_objects_per_image,
        )
        # transform image
        trans_img, trans_bounding_boxes = self.fit_transform(image, bounding_boxes)

        return (
            trans_img,
            trans_bounding_boxes,
            torch.tensor(
                [[self.label_to_idx.get(label)] for label in labels],
                dtype=torch.int8,
            ),
            image.shape[0] / self.aim_height,
            image.shape[1] / self.aim_width,
        )


# %%
image_config = {
    "train": {
        idx: {
            "image_path": os.path.join(
                "./data/archive/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
                img_name + ".jpg",
            ),
            "annotation_path": os.path.join(
                "./data/archive/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
                img_name + ".xml",
            ),
        }
        for idx, img_name in enumerate(
            open(
                file="./data/archive/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Layout/train.txt",
                mode="r",
            )
            .read()
            .split("\n")
        )
        if img_name != ""
    },
    "val": {
        idx: {
            "image_path": os.path.join(
                "./data/archive/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
                img_name + ".jpg",
            ),
            "annotation_path": os.path.join(
                "./data/archive/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
                img_name + ".xml",
            ),
        }
        for idx, img_name in enumerate(
            open(
                file="./data/archive/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Layout/val.txt",
                mode="r",
            )
            .read()
            .split("\n")
        )
        if img_name != ""
    },
    "test": {
        idx: {
            "image_path": os.path.join(
                "./data/archive/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
                img_name + ".jpg",
            ),
            "annotation_path": os.path.join(
                "./data/archive/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
                img_name + ".xml",
            ),
        }
        for idx, img_name in enumerate(
            open(
                file="./data/archive/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Layout/test.txt",
                mode="r",
            )
            .read()
            .split("\n")
        )
        if img_name != ""
    },
}

# %%
pvid = PascalVOCImageDataset(
    image_config=image_config,
    source="train",
    do_transform=False,
    is_inception_backbone=False,
)
# %%
img, bboxes, labels, scale_height, scale_width = next(
    iter(
        DataLoader(pvid, batch_size=64, shuffle=True),
    ),
)
# %%
plt.imshow(img[0].permute(1, 2, 0).numpy())
# %%
print("Min value:", np.min(img[0].numpy()))
print("Max value:", np.max(img[0].numpy()))
# %%
image_config["train"][0]["image_path"]
# %%
