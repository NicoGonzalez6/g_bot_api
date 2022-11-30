import torch



# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# # Images

BASE_URL = "./data/images/"
FILE_NAMES = ['2.png']

imgs = [BASE_URL + file_name for file_name in FILE_NAMES]

# # # Inference
# results = model(dir)
# print(results.xyxy[0])

# Inference
results = model(imgs)

# Display the results
#results.show()

# Save the results
#results.save(save_dir='runs/detect/exp')



# Print the results
print("Results")
results.print()


print(len(results.pandas().xyxy[0]),'vainas')



# for i, file_name in enumerate(FILE_NAMES):
#     print(file_name)
#     print(results.xyxy[i])
#     print()