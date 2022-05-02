from matplotlib import widgets
import wget

train_img_url = "http://images.cocodataset.org/zips/train2017.zip"
val_img_url = "http://images.cocodataset.org/zips/val2017.zip"
train_annot_url = "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip"

print(f"Downloading from {train_img_url}")
wget.download(train_img_url)

print()

print(f"Downloading from {val_img_url}")
wget.download(val_img_url)

print()

print(f"Downloading from {train_annot_url}")
wget.download(train_annot_url)