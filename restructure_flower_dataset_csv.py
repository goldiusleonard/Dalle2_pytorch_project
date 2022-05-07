from csv import writer
from pathlib import Path
import os
import pandas as pd

# Change your "ImagesCombine" folder path here
img_root_path = "./Flower_Dataset_Combine/ImagesCombine/"

# Change your flower dataset csv path here
csv_path = "./Flower_Dataset_Combine/Captions.csv"

# Change your new flower dataset csv save path here
new_csv_path = "./Flower_Dataset_Combine/New_captions.csv"

old_csv = pd.read_csv(csv_path, header=None)

with open(new_csv_path, 'w', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(['file_name', 'caption'])

    for idx, sub_header in enumerate(sorted(Path(img_root_path).iterdir())):
        caption = old_csv.loc[idx][0]
        file_name = os.path.basename(sub_header)

        csv_writer.writerow([file_name, caption])