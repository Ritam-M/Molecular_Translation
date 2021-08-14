# Molecular_Translation

Given an image of a molecule, our goal is to predict the InCHI string for that molecule. We use Image Captioning Ideas for the same, where we use various combinations of 
EfficientNet, ResNet and Visual Transformers as Encoder and Multi-Step LSTM and Transformers as Decoder. We use Levenshtein Distance as our metric. This Repo consists code of EfficientNet+Transformer.

| Methods | Levenshtein Distance |
|---------|----------------------|
|ResNet+Multi-layered-LSTM| 3.97|
|EfficientNet+Multilayered-LSTM| 3.24|
|Efficient+Transformer|2.39| 
|Vision-Transformer+Transformer|3.11|
|Efficient-Ensemble+Transformer|1.31|

Run the following commands before implementing:
pip install -q --upgrade pip
pip install -q efficientnet
