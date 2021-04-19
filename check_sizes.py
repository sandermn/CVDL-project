from medimage import image
from pathlib import Path

base_path = Path('/work/datasets/medical_project/CAMUS')

im = image(base_path/"patient0001/patient0001_4CH_ED.mhd")

print(im.size())