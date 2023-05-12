# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils
import argparse



FILE_PATH = "birth_dev.tsv"
f = open(FILE_PATH, "r")
n_pred = len(f.readlines())

pred = ["London"] * n_pred

print(utils.evaluate_places(FILE_PATH, pred))
