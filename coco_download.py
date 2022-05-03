import wget

train_img_url = "http://images.cocodataset.org/zips/train2014.zip"
val_img_url = "http://images.cocodataset.org/zips/val2014.zip"
train_annot_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

print(f"Downloading from {train_img_url}")
wget.download(train_img_url)

print()

print(f"Downloading from {val_img_url}")
wget.download(val_img_url)

print()

print(f"Downloading from {train_annot_url}")
wget.download(train_annot_url)