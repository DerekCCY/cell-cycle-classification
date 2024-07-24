import splitfolders

input_folder1 = "/home/ccy/cellcycle/data/FUCCI"
output_folder1= "/home/ccy/cellcycle/data/FUCCI/Split"
splitfolders.ratio(input_folder1, output=output_folder1, seed=42, ratio=(0.7, 0.25, 0.05))