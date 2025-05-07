import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import train_dataloader, val_dataloader, test_dataloader
from model import CNNLSTMModel
from sklearn.metrics import accuracy_score
import numpy as np

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 27  
    model = CNNLSTMModel(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0001  
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # RMSprop da dene (optimizer = optim.RMSprop(model.parameters(), lr=learning_rate))

    num_epochs = 5
    best_val_accuracy = 0.0
    early_stopping_patience = 5
    no_improvement_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []
        correct_predictions = 0
        total_samples = 0

        for i, (inputs, labels, lengths) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()

            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            accuracy = correct_predictions / total_samples

            if i % 10 == 9:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {running_loss / 10:.4f}, Accuracy: {accuracy:.4f}')
                running_loss = 0.0

        train_accuracy = accuracy_score(all_labels, all_predictions)

        model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_predictions = []
        val_correct_predictions = 0
        val_total_samples = 0

        with torch.no_grad():
            for inputs, labels, lengths in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                lengths = lengths.to(device)

                outputs = model(inputs, lengths)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_predictions.extend(predicted.cpu().numpy())

                val_correct_predictions += (predicted == labels).sum().item()
                val_total_samples += labels.size(0)

        val_accuracy = val_correct_predictions / val_total_samples
        avg_val_loss = val_loss / len(val_dataloader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Erken durdurma (Early Stopping)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model3.pth')
            print("Best model saved.")
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print('Finished Training')

    test_correct_predictions = 0
    test_total_samples = 0
    all_test_labels = []
    all_test_predictions = []

    with torch.no_grad():
        for inputs, labels, lengths in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            lengths = lengths.to(device)

            outputs = model(inputs, lengths)
            _, predicted = torch.max(outputs.data, 1)
            all_test_labels.extend(labels.cpu().numpy())
            all_test_predictions.extend(predicted.cpu().numpy())

            test_correct_predictions += (predicted == labels).sum().item()
            test_total_samples += labels.size(0)

    test_accuracy = test_correct_predictions / test_total_samples
    print(f'Test Accuracy: {test_accuracy:.4f}')

if __name__ == '__main__':
    train_model()