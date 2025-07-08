import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim : int = 512, reduction_ratio : int = 16):
        super(CrossModalAttention, self).__init__()
        
        self.channel_reduction1 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // reduction_ratio, feature_dim, kernel_size=1, bias=False)
        )
        
        self.channel_reduction2 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // reduction_ratio, feature_dim, kernel_size=1, bias=False)
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.fusion_conv = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights using Xavier initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def channel_attention(self, x):

        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        max_pool = F.adaptive_max_pool2d(x, (1, 1))
        
        avg_out = self.channel_reduction1(avg_pool)
        max_out = self.channel_reduction2(max_pool)
        
        channel_att = torch.sigmoid(avg_out + max_out)
        
        return x * channel_att
    
    def spatial_att(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]        
        spatial_cat = torch.cat([avg_out, max_out], dim=1)        
        spatial_att = self.spatial_attention(spatial_cat)
        
        return x * spatial_att
    
    def forward(self, x1, x2):

        x1_channel_att = self.channel_attention(x1)
        x2_channel_att = self.channel_attention(x2)
        
        x1_spatial_att = self.spatial_att(x1_channel_att)
        x2_spatial_att = self.spatial_att(x2_channel_att)
        
        fused_features = torch.cat([x1_spatial_att, x2_spatial_att], dim=1) 
        output = self.fusion_conv(fused_features)
        
        return output
    


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, 
                 feature_dim: int = 512, 
                 reduction_ratio: int = 16, 
                 mlp_ratio: int = 4, 
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.cross_modal_attention = CrossModalAttention(
            feature_dim=feature_dim, 
            reduction_ratio=reduction_ratio
        )
        
        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(feature_dim, mlp_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(mlp_hidden_dim, feature_dim, kernel_size=1),
            nn.Dropout(dropout_rate)
        )
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x1, x2):

        x1_norm = self.norm1(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x2_norm = self.norm1(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        x_cross = self.cross_modal_attention(x1_norm, x2_norm)
        
        x_cross = x1 + self.dropout(x_cross)
        
        x_norm = self.norm2(x_cross.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_cross = x_cross + self.dropout(self.mlp(x_norm))
        
        return x_cross




"""cross_modal_attn = CrossModalAttention()
cross_attn_block = CrossModalAttentionBlock()

x1 = torch.randn(3, 512, 32, 32) 
x2 = torch.randn(3, 512, 32, 32) 
    
output = cross_modal_attn(x1, x2)
print("Input shapes:", x1.shape, x2.shape)
print("Output shape:", output.shape)
output = cross_attn_block(x1, x2)
print("Output shape:", output.shape)"""



class GroupQueryAttention(nn.Module):
    def __init__(self, 
                 feature_dim : int = 512, 
                 num_groups : int = 8, 
                 num_heads : int = 4, 
                 dropout_rate : float = 0.1):

        super(GroupQueryAttention, self).__init__()
        
        assert feature_dim % num_groups == 0, "feature_dim must be divisible by num_groups"
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.feature_dim = feature_dim
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        self.group_query = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim // num_groups, self.head_dim, kernel_size=1),
                nn.BatchNorm2d(self.head_dim),
                nn.GELU()
            ) for _ in range(num_groups)
        ])
        
        self.group_key = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim // num_groups, self.head_dim, kernel_size=1),
                nn.BatchNorm2d(self.head_dim),
                nn.GELU()
            ) for _ in range(num_groups)
        ])
        
        self.group_value = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim // num_groups, self.head_dim, kernel_size=1),
                nn.BatchNorm2d(self.head_dim),
                nn.GELU()
            ) for _ in range(num_groups)
        ])
        
        self.group_fusion = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )

        self.group_fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),  # Changed input channels
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )
        
        self.scale = self.head_dim ** -0.5
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x):

        batch_size, _, height, width = x.size()
        
        x_groups = torch.chunk(x, self.num_groups, dim=1)
        
        group_attentions = []
        
        for group_idx in range(self.num_groups):
            group_x = x_groups[group_idx]
            
            query = self.group_query[group_idx](group_x)
            key = self.group_key[group_idx](group_x)
            value = self.group_value[group_idx](group_x)
            
            query = query.view(batch_size, self.head_dim, height * width).transpose(1, 2)
            key = key.view(batch_size, self.head_dim, height * width)
            value = value.view(batch_size, self.head_dim, height * width).transpose(1, 2)
            
            attention_scores = torch.matmul(query, key) * self.scale
            attention_probs = F.softmax(attention_scores, dim=-1)
            
            attention_probs = self.dropout(attention_probs)
            
            group_attended = torch.matmul(attention_probs, value)
            group_attended = group_attended.transpose(1, 2).contiguous()
            group_attended = group_attended.view(batch_size, self.head_dim, height, width)
            
            group_attentions.append(group_attended)
        
        group_output = torch.cat(group_attentions, dim=1)
        
        output = self.group_fusion(group_output)
        output = self.layer_norm(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return output


class GroupedAttentionBlock(nn.Module):
    def __init__(self, 
                 feature_dim : int = 512, 
                 num_groups : int = 8, 
                 num_heads : int = 4, 
                 mlp_ratio : int = 4, 
                 dropout_rate : float = 0.1):

        super().__init__()
        
        self.group_query_attention = GroupQueryAttention(
            feature_dim=feature_dim, 
            num_groups=num_groups, 
            num_heads=num_heads, 
            dropout_rate=dropout_rate
        )
        
        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(feature_dim, mlp_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(mlp_hidden_dim, feature_dim, kernel_size=1),
            nn.Dropout(dropout_rate)
        )
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):

        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.dropout(self.group_query_attention(x_norm))
        
        x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.dropout(self.mlp(x_norm))
        
        return x



"""group_query_attn = GroupQueryAttention(feature_dim=512, num_groups=8, num_heads=4)
grouped_attn_block = GroupedAttentionBlock(feature_dim=512, num_groups=8, num_heads=4)
    
x = torch.randn(3, 512, 32, 32)
    
gqa_output = group_query_attn(x)
print("GroupQueryAttention Input shape:", x.shape)
print("GroupQueryAttention Output shape:", gqa_output.shape)
    
gab_output = grouped_attn_block(x)
print("GroupedAttentionBlock Input shape:", x.shape)
print("GroupedAttentionBlock Output shape:", gab_output.shape)"""