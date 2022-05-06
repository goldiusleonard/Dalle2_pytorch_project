import wget

train_img_url = "https://ivc.ischool.utexas.edu/VizWiz_final/images/train.zip"
val_img_url = "https://ivc.ischool.utexas.edu/VizWiz_final/images/val.zip"
test_img_url = "https://ivc.ischool.utexas.edu/VizWiz_final/images/test.zip"
annot_url = "http://ivc.ischool.utexas.edu/VizWiz_final/caption/annotations.zip"

print(f"Downloading from {train_img_url}")
wget.download(train_img_url)

print()

print(f"Downloading from {val_img_url}")
wget.download(val_img_url)

print()

print(f"Downloading from {test_img_url}")
wget.download(test_img_url)

print()

print(f"Downloading from {annot_url}")
wget.download(annot_url)