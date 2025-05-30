from ultralytics import YOLO
import cv2
import os

model = YOLO('my_model.pt')
image_folder = 'input_images'
output_folder = 'cropped_outputs'
os.makedirs(output_folder, exist_ok=True)

results = model(image_folder)  

for result in results:
    image_path = result.path
    img = cv2.imread(image_path)
    boxes = result.boxes.xyxy #[x1,y1,x2,y2]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        crop = img[y1:y2, x1:x2]
        crop_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_crop_{i}.jpg"
       
       #store it to local 
        cv2.imwrite(os.path.join(output_folder, crop_name), crop)

print(f"Saved crops to {output_folder}")


# get page no. with primary key from mongoDb
# get image_url
# pass image to custom trained YOLO model
# Get predicted Cordinates
# crop predicted cordinates
# upload cropped images to google cloud it will return image_urls
# update image_urls to respective page no. 
