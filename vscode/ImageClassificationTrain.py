import os
import torch
from torchvision import datasets, models, transforms
import torch.onnx as onnx

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataDirPath = R"R:\testImg_Line224" 
    image_datasets = datasets.ImageFolder(dataDirPath, data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=12, shuffle=True, num_workers=16)

    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, 128)
    
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  
        for inputs, labels in dataloaders:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    onnx_file_path = os.path.join(dataDirPath, R"R:\testImg_Line224\model.onnx")
    torch.onnx.export(model, torch.randn(1, 3, 224, 224).to(device), onnx_file_path,input_names=["inputImage"],output_names=["outputArray"])
