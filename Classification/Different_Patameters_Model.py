"""
    目的： 利用creat_writer(), train_update() 控制结果存储

    过程： 通过上述函数来跑不同参数的模型。 不同参数： epochs, model类型， 训练的datasets
    # 5. 跑不同参数的数据model： epochs, model类型， 训练的数据量
    # 6. 将5中的model进行tensorboard查看评估。 评估可以考虑Fn分数， recall值等参数
    # 7. 使用最好的模型进行最后的预测
    # 8. 对 ‘模型的结果’ 和 ‘模型本身的训练后参数’ 进行保存
"""

import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

from torchinfo import summary

from pathlib import Path

from helper_func import data_setup, plot_loss_curves, device, pred_and_plot_image

from Tracking import train_update, create_writer


# 1. 得到不同数量的数据
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # values per colour channel [red, green, blue]
                                 std=[0.229, 0.224, 0.225]) # values per colour channel [red, green, blue]

simple_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 1. Resize the images
    transforms.ToTensor(), # 2. Turn the images into tensors with values between 0 & 1
    normalize # 3. Normalize the images so their distributions match the ImageNet dataset
])

# 数据量为10% & 20%
BATCH_SIZE = 32

train_dataloader_10_percent, test_dataloader, class_names = data_setup(train_dir="data/pizza_steak_sushi/train",
    test_dir="data/pizza_steak_sushi/test",
    transform=simple_transform,
    batch_size=BATCH_SIZE
)

# 只需要训练数据变大即可， 不管测试数据
train_dataloader_20_percent, test_dataloader, class_names = data_setup(train_dir="data/pizza_steak_sushi_20_percent/train",
    test_dir="data/pizza_steak_sushi/test",
    transform=simple_transform,
    batch_size=BATCH_SIZE
)


# 2. 引入不同的model : 开始时只考虑了引入， 现在开始封装
OUT_FEATURES = len(class_names)

def create_effnetb0():
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    torch.manual_seed(42)
    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=OUT_FEATURES)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb0"
    print(f"[INFO] Created new {model.name} model.")
    return model

def create_effnetb2():
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    torch.manual_seed(42)
    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1408, out_features=OUT_FEATURES)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb2"
    print(f"[INFO] Created new {model.name} model.")
    return model

"""
    观察到： 我们的不同参数为： epochs, model, dataset。 所以做了：
    1. model构建封装 
    2. epochs用List表示：[5, 10]  
    3. dataset创建了10%，20%的版本
"""

if __name__ == "__main__" :
    # 2。2 ： 实例化model
    effnet_b0 = create_effnetb0()
    effnet_b2 = create_effnetb2()
    # 可以用torchinfo.summary进行结构查看

    # 3. 训练
    num_epochs = [5, 10]
    models = ["effnetb0", "effnetb2"]
    train_dataloaders = {"train_dataloader_10_percent": train_dataloader_10_percent,
                         "train_dataloader_20_percent": train_dataloader_20_percent}


    # 注意后续的方法， 通过for循环遍历了所有模型的情况， 保证代码的简洁度
        # 前面做的一切封装和List,Dict都是为了这下面的for循环遍历服务
    torch.manual_seed(42)
    torch.mps.manual_seed(42)
    experiment_number = 0

    for dataloader_name, train_dataloader in train_dataloaders.items():

        for epochs in num_epochs:

            for model_name in models:

                experiment_number += 1  # 事实查看训练到了哪个期望模型
                print(f"[INFO] Experiment number: {experiment_number}")
                print(f"[INFO] Model: {model_name}")
                print(f"[INFO] DataLoader: {dataloader_name}")
                print(f"[INFO] Number of epochs: {epochs}")

                if model_name == "effnetb0":
                    model = create_effnetb0()
                else:
                    model = create_effnetb2()

                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

                train_update(model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=epochs,
                      device=device,
                      writer=create_writer(experiment_name=dataloader_name,
                                           model_name=model_name,
                                           extra=f"{epochs}_epochs"))   # writer的保存信息与模型的三个参数相关，方便管理查看。 此处根据需要自定义

                # 10. save model.stat_dict()
                save_file_name = f"07_{model_name}_{dataloader_name}_{epochs}_epochs.pth"    # 模型的name,
                save_filepath = Path("models") / save_file_name
                torch.save(obj=model.state_dict(), f=save_filepath)
                print("-" * 50 + "\n")


    # 4. evaluate: 可视化传入一些照片看结果。 在bestModel文件中实现

    """
        summary : 
            1. 我们很好地实现了目录简便化的管理， 可以自定义这部分。 详细参考create_writer()函数。 根据初始化writer来自定义
            2. 通过1来方便可视化我们的model的性能表现， 更好更直观地看到我们的评估图
            3. 将model存储
    """