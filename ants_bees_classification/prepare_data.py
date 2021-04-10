# Download and unzip data (google drive)
import os, urllib, zipfile

path = '/content/drive/MyDrive/Colab Notebooks/DeepLearningWithPyTorch'
os.chdir(path)

data_dir = './data'
url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
save_path = os.path.join(data_dir,'hymenoptera_data.zip')

if not os.path.exists(os.path.join(data_dir,'hymenoptera_data')):
    urllib.request.urlretrieve(url,save_path)

    # read by zipfile
    zip = zipfile.ZipFile(save_path)
    zip.extractall(data_dir)
    zip.close()

    os.remove(save_path)

# Check size of each directory

# !ls data/hymenoptera_data/train/ants | wc -l

# !ls data/hymenoptera_data/train/bees | wc -l

# !ls data/hymenoptera_data/val/ants | wc -l

# !ls data/hymenoptera_data/val/bees | wc -l