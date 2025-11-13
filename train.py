import modal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from zoneinfo import ZoneInfo
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import random_split
from model import ImageCNN

app = modal.App("CatsAndDogs")

# Creating the image and using the cli to set up training
image = (modal.Image.debian_slim() # Creating an image of a linux debian installation 
         .pip_install_from_requirements("Coding Projects/CatsVsDogs/requirements.txt") # Used to install requirements.txt in this dir
         .add_local_python_source("model") # Import the model library from this dir
         .add_local_dir("Coding Projects/CatsVsDogs/data", "/CatsVsDogs_Data")) 

volume = modal.Volume.from_name("cats_and_dogs_data", create_if_missing=True) # Creates the Volume used to store tensorboard data 
model_volume = modal.Volume.from_name("cats_and_dogs_model", create_if_missing=True) # Creates the volume needed 

# Shift + Tab to remove one space of indendation don't forget!



@app.function(image=image, gpu="A10G", volumes = {"/data": volume, "/models": model_volume}, timeout= 60 * 60 * 3)
def train():
    from datetime import datetime

    
    base_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    train_dataset = ImageFolder(
        root="/CatsVsDogs_Data/train",
        transform=train_transform
    )

    val_dataset = ImageFolder(
        root="/CatsVsDogs_Data/val", 
        transform=base_transform
    )

    training_data = train_dataset
    test_data = val_dataset

    time_hk = datetime.now(ZoneInfo("Asia/Shanghai"))
    timestamp = time_hk.strftime("%Y%m%d_%H%M%S")
    log_dir = f"/models/tensorboard_logs/run_{timestamp}"
    writer = SummaryWriter(log_dir)


    print(f"Training Samples: {len(training_data)}")
    print(f"Test Samples: {len(test_data)}")
    print(f"Dataset classes: {len(train_dataset.classes)}")

    train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)

    num_epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageCNN(input_shape=3, num_classes=len(train_dataset.classes))
    model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 1e-2)

    scheduler = OneCycleLR(
        optimizer,
        max_lr = 1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.3
    )

    best_acc = 0.0

    print("Starting Training")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            train_dataloader, desc=f'Epoch: {epoch+1}/{num_epochs}')
        for data, target in progress_bar:
            data, target = data.to(device), target.long().to(device)

            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})    

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # Do validation after each epoch
        model.eval()

        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.long().to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(test_dataloader)

        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

        print(f"Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes
            }, '/models/best_model.pth')
            print(f'New best model saved: {accuracy:.2f}%')

    writer.close()
    print(f'Training Complete! Best Accuracy: {best_acc:.2f}%')


@app.local_entrypoint()
def main():
    train.remote()
