# A Cost Effective Framework to Predict the Ripeness of any Fruit Based on Color Spread

## Citation Information
This repository contains the code and dataset used for the paper given below. Please cite the paper if you use the code in this repository as part of a published research project.
*A. Mueez, “A Cost-Effective Framework to Predict the Ripeness of any Fruit Based on Color Space”, 2020 IEEE Region 10 Symposium (TENSYMP), Dhaka, Bangladesh. In press.*


## A brief outline of the contents:

* **Raspberry Pi**: This folder contains files that have been written to run on a Raspberry Pi equipped with openCV, scikit-learn and pandas library:
  * **tain.py**: With this program running when a fruit is placed on the rotating platform, the sonar detects it and asks the user to enter an appropirate ripeness index for it. Then five images are captured with the servo rotating 72 degrees after each capture. Simultaneously, for each image captured, values for the following attributes -	*hue1,	sat1,	val1,	hue2,	sat2,	val2,	hue3,	sat3,	val3,	ripenessIndex* are written to a CSV file.
  * **test.py**: Similary, when this program is run and a fruit is placed on the podium, using the previously generated dataset, the program predicts a ripeness value. Based on whether this value is even or odd, the servo attached on either side of the structure pushes the fruit towards the left or the right.
  

* **Dataset (Images)**: This folder contains 273 images of Egyptian banana species obtained from this [paper](https://link.springer.com/article/10.1007/s13369-018-03695-5) along with their resized versions.

* **Dataset (CSV)**: This folder contains the HSV values generated from the resized images.

* **CSVDatasetFromImages.py**: This program generates the top 3 most dominant HSV values from each image contained within a folder and writes the values back to a CSV file. User has to specify three things:
  * Folder directory
  * CSV file name (generated by the program)
  * Ripeness level of all the fruits within that folder

* **ConcatDataframes.py**: This program concatenates the distinct CSV files produced for each class into one. User needs to specify *file_name_suffix* which should be either *train* or *test*. The output of this program, the merged files, are present in the *Dataset (CSV)* folder.
 
* 	**KNN Results**: This Jupyter Notebook file contains metrics regarding the accuracy of K Nearest Neighbors algorithm when applied on the dataset generated.

## Segmentation issues:
Images, *m013* and *v026* were segmneted inaccurately. This is reflected in the *test.csv* file as anomalous values.
