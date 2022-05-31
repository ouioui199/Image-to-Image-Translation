import os
import csv
import cv2

path_to_read_image = "C:\\Stage5A\\GANs\\Pix2Pix\\dataset\\img_align_celeba\\img_align_celeba"
path_to_write_image = "C:\\Stage5A\\GANs\\Pix2Pix\\dataset\\img_align_celeba\\img_align_celeba_with_blackbox"

if os.path.exists(path_to_write_image) == False:
    os.makedirs(path_to_write_image, exist_ok=True)
else:
    print("Writing folder exists and may contain folders! ")
    quit()

path_to_csv = os.path.dirname(path_to_read_image) + "\\list_landmarks_align_celeba.csv"

with open(path_to_csv, mode='r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if int(row['image_id'].strip('.jpg')) > 125250: # To resume from the last work
            image_file = path_to_read_image + "\\" + row['image_id']
            image = cv2.imread(image_file)

            lefteye_x = row['lefteye_x']
            lefteye_y = row['lefteye_y']
            righteye_x = row['righteye_x']
            righteye_y = row['righteye_y']

            lefteye_x = int(lefteye_x) - 15
            lefteye_y = int(lefteye_y) - 10
            righteye_x = int(righteye_x) + 15
            righteye_y = int(righteye_y) + 10

            start_point = (lefteye_x, lefteye_y)
            end_point = (righteye_x, righteye_y)

            color = (0, 0, 0)
            thickness = -1

            image = cv2.rectangle(image, start_point, end_point, color, thickness)
 
            cv2.imwrite(path_to_write_image + "\\" + os.path.basename(image_file) + "_blackbox.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    
            print("Finished writing image:", os.path.basename(image_file))
