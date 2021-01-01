import zipfile
with zipfile.ZipFile('mmsegmentation/data/cityscapes/leftImg8bit_trainvaltest.zip', 'r') as zip_ref:
    zip_ref.extractall('mmsegmentation/data/cityscapes/')
with zipfile.ZipFile('mmsegmentation/data/cityscapes/gtFine_trainvaltest.zip', 'r') as zip_ref:
    zip_ref.extractall('mmsegmentation/data/cityscapes/')