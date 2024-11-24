import os
import numpy as np
import sys
import random
from PIL import Image

if len(sys.argv) < 2:
    print("Format : python " + sys.argv[0] + " <Nombre de dossiers max>")
    sys.exit(1)

images = []
max_img = int(sys.argv[1])

i = 1
data_dir = "/mnt/c/Users/Valentin/Documents/temp/lfw/lfw"
ddir = os.listdir(data_dir)
random.shuffle(ddir)
for person_name in ddir:
    if i > max_img:
        break

    person_path = os.path.join(data_dir, person_name)
    if not os.path.isdir(person_path):
        continue
    
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        
        try:
            img = Image.open(image_path).convert('RGB')
            img_resized = img.resize((256, 256))
            img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
            
            images.append(img_array)
        except Exception as e:
            print(f"Erreur avec {image_path}: {e}")

    print("Chargement du dossier", i, "/", min(max_img, len(ddir)))
    i += 1

images = np.array(images)

np.save("../datasets/lfw-dataset.npy", images)