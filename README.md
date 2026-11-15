```markdown
# CIFAR-100 Classification using Deep Transfer Learning with ResNet-50

This project implements a highly optimized PyTorch pipeline to classify the **CIFAR-100** dataset. By utilizing deep transfer learning with a pre-trained **ResNet-50** backbone, upscaling input images to $224 \times 224$ pixels, applying advanced data augmentation, and regularizing with Label Smoothing, the model achieves a state-of-the-art **84.95% validation accuracy**.

## 🚀 Features
- **Pre-trained Backbone**: Fine-tuned PyTorch ImageNet pre-trained `ResNet-50` model.
- **Input Upscaling**: Resizes $32 \times 32$ CIFAR-100 images to $224 \times 224$ to leverage native resolution features of deep networks.
- **Advanced Augmentation**: Applies random horizontal flips and random cropping to prevent overfitting.
- **Regularization & Optimization**:
  - **Label Smoothing Loss**: Prevents overconfidence during training.
  - **Cosine Annealing Scheduler**: Anneals learning rate dynamically across 10 training epochs.

---

## 📊 Results and Performance
Our optimal configuration achieved the following metrics on the validation subset:

| Epoch | Train Loss | Validation Loss | Validation Accuracy |
| :---: | :--------: | :-------------: | :-----------------: |
| 1     | 2.0378     | 1.6679          | 71.62%              |
| 5     | 1.1024     | 1.3672          | 81.13%              |
| 8     | 0.8751     | 1.3030          | 83.62%              |
| **10**| **0.8299** | **1.2725**      | **84.95%**          |

---

## 🛠️ Project Structure
```bash
├── models/
│   └── cifar100_resnet50.pth    # Saved PyTorch State Dict weights
├── helper_functions.py          # PyTorch evaluation & metric utilities
├── cifar100_resnet50.ipynb     # Main training notebook
└── README.md                    # Project Documentation
```

---

## 🔧 Installation & Getting Started

### Prerequisites
Make sure you have Python 3.8+ and PyTorch installed with GPU enablement:
```bash
pip install torch torchvision datasets scikit-learn matplotlib tqdm
```

### Load and Inference
To use the saved weights (`cifar100_resnet50.pth`) inside your application:

```python
import torch
import torchvision.models as models
import torch.nn as nn

class CIFAR100_PretrainedResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.resnet = models.resnet50(weights=None) 
        self.resnet.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        return self.resnet(x)

# Instantiate model and load weights
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CIFAR100_PretrainedResNet50(num_classes=100).to(device)
model.load_state_dict(torch.load("models/cifar100_resnet50.pth", map_location=device))
model.eval()
print("Model loaded successfully!")
```

---

## 👥 Authors
- Your Name / GitHub username ([@Yubraj45](https://github.com/Yubraj45))
```
