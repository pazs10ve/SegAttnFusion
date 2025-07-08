import torch
import torch.nn as nn
from src.cnn_encoder import DenseNet121
from src.word_decoder import WordPredictionTransformer
from src.segmentation import MedicalImageSegmentor, combine
from inference.upscale import upscale
from src.attention import CrossModalAttentionBlock, GroupedAttentionBlock


config_dict = {
    'encoders': {'densenet121': DenseNet121},
    'decoders': {'pubmedbert': WordPredictionTransformer},
    'segmenters': {'deeplabv3+': MedicalImageSegmentor},
    'super_res': {'ESRGAN': upscale}
}


def get_model(
        device : str,
        encoder: str = 'densenet121',
        decoder: str = 'pubmedbert',
        segmenter: str = 'deeplabv3+',
        super_res_method: str = 'ESRGAN',
        cross_attn_blocks: int = 6,
        grp_attn_blocks: int = 6,
):

    encoder_model = config_dict['encoders'][encoder]()
    decoder_model = config_dict['decoders'][decoder]()
    segmenter_model = config_dict['segmenters'][segmenter]()
    super_res_model = config_dict['super_res'][super_res_method]

    cross_attention_blocks = torch.nn.ModuleList([
        CrossModalAttentionBlock() for _ in range(cross_attn_blocks)
    ])

    grouped_attention_blocks = torch.nn.ModuleList([
        GroupedAttentionBlock() for _ in range(grp_attn_blocks)
    ])

    flatten = nn.Flatten()
    projection = nn.Linear(in_features = 512*8*8, out_features=1024)

    class CombinedMedicalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder_model
            self.decoder = decoder_model
            self.segmenter = segmenter_model
            self.super_res = super_res_model
            self.cross_attention = cross_attention_blocks
            self.grouped_attention = grouped_attention_blocks

            self.encoder.to(device)
            self.decoder.to(device)
            self.segmenter.to(device)
            #self.super_res.to(device)
            self.cross_attention.to(device)
            self.grouped_attention.to(device)

            projection.to(device)

        
        def forward(self, x):
            features1 = self.encoder(x)    
            segmentation_masks = self.segmenter.predict(x)   
            combined_images = combine(x, segmentation_masks)    
            features2 = self.encoder(combined_images.float().to(device))
            
            for cross_attn in self.cross_attention:
                features = cross_attn(features1, features2)

            for grp_attn in self.grouped_attention:
                features = grp_attn(features)
            
            features = flatten(features)
            features = projection(features)
            logits = self.decoder(features)
            
            return logits

    return CombinedMedicalModel()


"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(device)
x = torch.randn(12, 3, 256, 256).to(device)
x = model(x)
print(x.shape)
"""






"""def get_model(
        device : str,
        encoder: str = 'densenet121',
        decoder: str = 'pubmedbert',
        segmenter: str = 'deeplabv3+',
        super_res_method: str = 'ESRGAN',
        cross_attn_blocks: int = 6,
        grp_attn_blocks: int = 6,
):

    encoder_model = config_dict['encoders'][encoder]()
    decoder_model = config_dict['decoders'][decoder]()
    segmenter_model = config_dict['segmenters'][segmenter]()
    super_res_model = config_dict['super_res'][super_res_method]

    cross_attention_blocks = torch.nn.ModuleList([
        CrossModalAttentionBlock() for _ in range(cross_attn_blocks)
    ])

    grouped_attention_blocks = torch.nn.ModuleList([
        GroupedAttentionBlock() for _ in range(grp_attn_blocks)
    ])

    flatten = nn.Flatten()
    projection = nn.Linear(in_features = 512*8*8, out_features=1024)

    class CombinedMedicalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder_model
            self.decoder = decoder_model
            self.segmenter = segmenter_model
            self.super_res = super_res_model
            self.cross_attention = cross_attention_blocks
            self.grouped_attention = grouped_attention_blocks

            self.encoder.to(device)
            self.decoder.to(device)
            self.segmenter.to(device)
            #self.super_res.to(device)
            self.cross_attention.to(device)
            self.grouped_attention.to(device)

            projection.to(device)

        
        
        def forward(self, x):
            print("input shape = ", x.shape)
            # First, encode initial features
            features1 = self.encoder(x)
            print("features 1 shape = ", features1.shape)
            
            # Pass through segmentation model
            segmentation_masks = self.segmenter.predict(x)
            print("seg masks shape = ", segmentation_masks.shape)
            
            # Combine original images with segmentation masks
            combined_images = combine(x, segmentation_masks)
            print("combined images shape = ", combined_images.shape)
            #combined_images.to(device)
            
            # Encode combined images to get second set of features
            features2 = self.encoder(combined_images.float().to(device))
            print("features 2 shape = ", features2.shape)
            
            # Pass features through cross-attention blocks
            for cross_attn in self.cross_attention:
                features = cross_attn(features1, features2)
            print("cross attn feature shape = ", features.shape)
            
            # Pass features through grouped attention blocks
            for grp_attn in self.grouped_attention:
                features = grp_attn(features)
            
            print("group attn feature shape = ", features.shape)

            features = flatten(features)
            print("features flatted = ", features.shape)

            features = projection(features)
            print("features projected", features.shape)

            logits = self.decoder(features)
            
            return logits

    return CombinedMedicalModel()"""