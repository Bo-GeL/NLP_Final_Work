from Different_Patameters_Model import create_effnetb2, pred_and_plot_image, class_names
import torch
from pathlib import Path


# 4. evaluate : 选择最好的model预测图片+随机自选图片
best_model_path = "models/07_effnetb2_train_dataloader_20_percent_10_epochs.pth"  # 加载最好model
best_model = create_effnetb2()
best_model.load_state_dict(torch.load(best_model_path))

# 开始传入图片来可视化预测结果
import random

num_images_to_plot = 3
test_image_path_list = list(
    Path("data/pizza_steak_sushi_20_percent/test").glob("*/*.jpg"))  # get all test image paths from 20% dataset
test_image_path_sample = random.sample(population=test_image_path_list,
                                       k=num_images_to_plot)  # randomly select k number of images

for image_path in test_image_path_sample:
    pred_and_plot_image(model=best_model,
                        image_path=image_path,
                        class_names=class_names,
                        image_size=(224, 224))

# Predict on custom image
pred_and_plot_image(model=best_model,
                    image_path="data/04-pizza-dad.jpeg",
                    class_names=class_names)
