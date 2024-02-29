"""
Script to perform train a model
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging
import os
import time
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import smdebug.pytorch as smd
except ModuleNotFoundError:
    print("[ERROR] Module 'smdebug' is not installed. Probably an inference container")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device, hook):
    hook.set_mode(smd.modes.EVAL)
    model.eval()
    running_loss = 0
    running_corrects = 0
    pred = []
    label = []


    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        
        logger.info(f"Prediction is:{preds}")
        logger.info(f"Label is: {labels.data}")
        
        new_pred = preds.tolist()
        new_label= labels.data.tolist()
        
        logger.info(f"Prediction List:{new_pred}")
        logger.info(f"Label List: {new_label}")
        
        pred.extend(new_pred)
        label.extend(new_label)
        
        logger.info(f"Final Prediction List Updated:{pred}")
        logger.info(f"Final Label List Updated: {label}")

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    
    metrics = {0:{"tp":0, "fp":0}, 1:{"tp":0, "fp":0}, 2:{"tp":0, "fp":0}, 3:{"tp":0, "fp":0}, 4:{"tp":0, "fp":0}}
    
    label_count = {0:0, 1:0, 2:0, 3:0, 4:0}
    for l, p in zip(label, pred):
        label_count[l]+=1
        if(p==l):
            metrics[l]["tp"]+=1
        else:
            metrics[p]["fp"]+=1
    
    logger.info(f"Metrics Computed: {metrics}")
    logger.info(f"Label Count Computed: {label_count}")
    
    F1 = {0:0, 1:0, 2:0, 3:0, 4:0}
    Precision = {0:0, 1:0, 2:0, 3:0, 4:0}
    Recall = {0:0, 1:0, 2:0, 3:0, 4:0}
    
    for c in Precision:
        denom = metrics[c]["tp"] + metrics[c]["fp"]
        if(denom==0):
            Precision[c]==0
        else:
            num = metrics[c]["tp"]
            Precision[c] = num/denom
            
    for c in Recall:
        denom = label_count[c]
        if(denom==0):
            Recall[c]==0
        else:
            num = metrics[c]["tp"]
            Recall[c] = num/denom
            
    for c in F1:
        if(Precision[c]==0 and Recall[c]==0):
            F1[c]=0
        else:
            num = 2*Precision[c]*Recall[c]
            denom = Precision[c] + Recall[c]
            F1[c] = num/denom
        
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    logger.info(f"Recall Computed: {Recall}")
    logger.info(f"F1 Computed: {F1}")


def train(model,
          train_loader,
          validation_loader,
          criterion,
          optimizer,
          device,
          hook,
          early_stopping):

    epochs = 10
    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0
    hook.set_mode(smd.modes.TRAIN)

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples += len(inputs)
                if running_samples % 100 == 0:
                    accuracy = running_corrects / running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0 * accuracy,
                        )
                    )

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1
            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase,
                                                                                        epoch_loss,
                                                                                        epoch_acc,
                                                                                        best_loss))
        if loss_counter == early_stopping:
            break
    return model


def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 128),
                   nn.ReLU(),
                   nn.Linear(128, 5))
    return model


def model_fn(model_dir):
    print("In model_fn. Model directory is -")
    print(model_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net().to(device)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading the dog classifier model")
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint)
        print('MODEL-LOADED')
        logger.info('model loaded successfully')
    model.eval()
    return model


def create_data_loaders(args, batch_size):
    train_data_path = os.path.join(args.data)
    test_data_path = os.path.join(args.test_dir)
    validation_data_path = os.path.join(args.valid_dir)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.ImageFolder(
        root=train_data_path,
        transform=train_transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    test_data = torchvision.datasets.ImageFolder(
        root=test_data_path,
        transform=test_transform
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    validation_data = torchvision.datasets.ImageFolder(
        root=validation_data_path,
        transform=test_transform,
    )
    validation_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_data_loader, test_data_loader, validation_data_loader


def main(args):
    logger.info(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}, Early Stopping: {args.early_stopping_rounds}')
    logger.info(f'Data Paths: {args.data}')

    model = net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.learning_rate)
    train_loader, test_loader, validation_loader = create_data_loaders(args, args.batch_size)
    model = model.to(device)
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    logger.info("Training the model")
    tic = time.perf_counter()
    model = train(model,
                  train_loader,
                  validation_loader,
                  criterion,
                  optimizer,
                  device,
                  hook,
                  early_stopping=args.early_stopping_rounds)
    toc = time.perf_counter()
    logger.info(f"Training took {toc - tic:0.4f} seconds")

    logger.info("Testing the model")
    test(model, test_loader, criterion, device, hook)

    logger.info("Saving the model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--early-stopping-rounds', type=int, default=10)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument("--valid_dir", type=str, default=os.environ["SM_CHANNEL_VALID"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()
    print(args)

    main(args)