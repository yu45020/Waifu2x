"""
Split images into small patches and insert them into sqlite db.  Reading and Inserting speeds are much better than
Ubuntu's (18.04) file system when the number of patches is larger than 20k. And it has smaller size than using h5 format

Recommend to check or filter out small size patches as their content vary little. 128x128 seems better than 64x64.


"""
import sqlite3
from torch.utils.data import DataLoader
from tqdm import trange
from Dataloader import Image2Sqlite

conn = sqlite3.connect('dataset/image_yandere.db')
cursor = conn.cursor()

with conn:
    cursor.execute("PRAGMA SYNCHRONOUS = OFF")

table_name = "train_images_size_128_noise_1_rgb"
lr_col = "lr_img"
hr_col = "hr_img"

with conn:
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({lr_col} BLOB, {hr_col} BLOB)")

dat = Image2Sqlite(img_folder='./dataset/yande.re_test_shrink',
                   patch_size=256,
                   shrink_size=2,
                   noise_level=1,
                   down_sample_method=None,
                   color_mod='RGB',
                   dummy_len=None)
print(f"Total images {len(dat)}")

img_dat = DataLoader(dat, num_workers=6, batch_size=6, shuffle=True)

num_batches = 20
for i in trange(num_batches):
    bulk = []
    for lrs, hrs in img_dat:
        patches = [(lrs[i], hrs[i]) for i in range(len(lrs))]
        # patches = [(lrs[i], hrs[i]) for i in range(len(lrs)) if len(lrs[i]) > 14000]

        bulk.extend(patches)

    bulk = [i for i in bulk if len(i[0]) > 15000]  # for 128x128, 14000 is fair. Around 20% of patches are filtered out
    cursor.executemany(f"INSERT INTO {table_name}({lr_col}, {hr_col}) VALUES (?,?)", bulk)
    conn.commit()

cursor.execute(f"select max(rowid) from {table_name}")
print(cursor.fetchall())
conn.commit()
# +++++++++++++++++++++++++++++++++++++
#           Used for Create Test Database
# -------------------------------------

# cursor.execute(f"SELECT ROWID FROM {table_name} ORDER BY LENGTH({lr_col}) DESC LIMIT 400")
# rowdis = cursor.fetchall()
# rowdis = ",".join([str(i[0]) for i in rowdis])
#
# cursor.execute(f"DELETE FROM {table_name} WHERE ROWID NOT IN ({rowdis})")
# conn.commit()
# cursor.execute("vacuum")
#
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS train_images_size_128_noise_1_rgb_small AS
# SELECT *
# FROM train_images_size_128_noise_1_rgb
# WHERE length(lr_img) < 14000;
# """)
#
# cursor.execute("""
# DELETE
# FROM train_images_size_128_noise_1_rgb
# WHERE length(lr_img) < 14000;
# """)

# reset index
cursor.execute("VACUUM")
conn.commit()

# +++++++++++++++++++++++++++++++++++++
#           check image size
# -------------------------------------
#

from PIL import Image
import io

cursor.execute(
    f"""
    select {hr_col} from {table_name} 
    ORDER BY LENGTH({hr_col}) desc 
    limit 100
"""
)
# WHERE LENGTH({lr_col}) BETWEEN 14000 AND 16000

# small = cursor.fetchall()
# print(len(small))
for idx, i in enumerate(cursor):
    img = Image.open(io.BytesIO(i[0]))
    img.save(f"dataset/check/{idx}.png")

# +++++++++++++++++++++++++++++++++++++
#           Check Image Variance
# -------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

dat = pd.read_sql(f"SELECT length({lr_col}) from {table_name}", conn)
dat.hist(bins=20)
plt.show()
