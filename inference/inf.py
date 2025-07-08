import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer

from model import ImageCaptioningModel


def load_model(model_path, vocab_size, embed_size, hidden_size, num_layers, device='cuda'):
    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def preprocess_image(image_path, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    return image, transform(image).unsqueeze(0)


def generate_caption(
    model, 
    image, 
    tokenizer, 
    max_length=256, 
    start_token=None, 
    end_token=None, 
    device='cuda'
):
    model.eval()
    image = image.to(device)
    
    with torch.no_grad():
        tokens = model.generate_caption(
            image, 
            max_length=max_length,
            start_token=start_token,
            end_token=end_token,
            device=device
        )
    
    caption = tokenizer.decode(tokens, skip_special_tokens=True)
    return caption


def inference(model_path, image_path, tokenizer_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'):
    VOCAB_SIZE = 30522
    EMBED_SIZE = 768
    HIDDEN_SIZE = 1024
    NUM_LAYERS = 12
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = load_model(
        model_path, 
        VOCAB_SIZE, 
        EMBED_SIZE, 
        HIDDEN_SIZE, 
        NUM_LAYERS, 
        DEVICE
    )
    
    original_image, processed_image = preprocess_image(image_path)
    
    caption = generate_caption(
        model, 
        processed_image, 
        tokenizer, 
        start_token=tokenizer.bos_token_id,
        end_token=tokenizer.eos_token_id
    )
    
    # Visualize the image and caption
    plt.figure(figsize=(10, 6))
    plt.imshow(original_image)
    plt.title(f"Generated Caption: {caption}", wrap=True)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return original_image, caption


model_path = r'model_checkpoint.pth'
image_path = r'data\images\CXR187_IM-0563\1.png'

# Run inference and display results
image, caption = inference(model_path, image_path)
print("\nCaption:", caption)

