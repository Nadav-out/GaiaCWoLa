import os
import sys
import tensorflow as tf

os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/global/home/users/njo/anaconda3/envs/gaia/lib/"

print("LD_LIBRARY_PATH set to:", os.environ["LD_LIBRARY_PATH"])

physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)
