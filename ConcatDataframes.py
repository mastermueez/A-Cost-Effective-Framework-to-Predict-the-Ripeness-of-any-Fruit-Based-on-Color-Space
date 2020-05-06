import pandas as pd
banana_classes = ["Green", "Yellowish_Green", "Midripen", "Overripen"]

column_names = ["file_name", "hue1", "hue2", "hue3", "ripeness_index"]
df = pd.DataFrame(columns = column_names)

file_name_suffix = "train.csv"
for banana_class in banana_classes:
    df_temp = pd.read_csv(banana_class+"_"+file_name_suffix)
    df = pd.concat([df, df_temp], axis=0)

df.to_csv(file_name_suffix, index=False)