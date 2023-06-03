import torch
from torch import nn


class SpatialTransformerEncoderP3(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=3):
        super(SpatialTransformerEncoderP3, self).__init__()

        # Define the positional encoding for the spatial dimensions
        h, w = 48, 80  # Height and width of the input images
        pos_enc_h = nn.Parameter(torch.randn(h, d_model // 2))
        pos_enc_w = nn.Parameter(torch.randn(w, d_model // 2))

        # Define the TransformerEncoder layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_layers)

        # Save the parameters
        self.pos_enc_h = pos_enc_h
        self.pos_enc_w = pos_enc_w

    def forward(self, x):
                                            
        batch_size, channels, height, width = x.size()

        # Compute the positional encodings for the spatial dimensions
        pos_emb_h = self.pos_enc_h.unsqueeze(1).repeat(1, width, 1)  # Shape: (height, width, d_model // 2)
        pos_emb_w = self.pos_enc_w.unsqueeze(0).repeat(height, 1, 1)  # Shape: (height, width, d_model // 2)
        pos_emb = torch.cat([pos_emb_h, pos_emb_w], dim=-1)  # Shape: (height, width, d_model)

        # Add the positional encoding to the input tensor
        x = x + pos_emb.permute(2, 0, 1).unsqueeze(0)  # Shape: (batch_size, channels, height, width, d_model)

        # Reshape the input tensor for processing by the TransformerEncoder
        x = x.permute(0, 3, 2, 1).reshape(batch_size * width, height, channels)

        # Apply the TransformerEncoder layer to each spatial location separately
        x = self.transformer_encoder(x)

        # Reshape the output tensor back to the original shape
        x = x.reshape(batch_size, width, height, channels, -1).permute(0, 3, 2, 1, 4)
        
        return x.squeeze()
    

class SpatialTransformerEncoderP4(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=3):
        super(SpatialTransformerEncoderP4, self).__init__()

        # Define the positional encoding for the spatial dimensions
        h, w = 24, 40  # Height and width of the input images
        pos_enc_h = nn.Parameter(torch.randn(h, d_model // 2))
        pos_enc_w = nn.Parameter(torch.randn(w, d_model // 2))

        # Define the TransformerEncoder layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_layers)

        # Save the parameters
        self.pos_enc_h = pos_enc_h
        self.pos_enc_w = pos_enc_w

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Compute the positional encodings for the spatial dimensions
        pos_emb_h = self.pos_enc_h.unsqueeze(1).repeat(1, width, 1)  # Shape: (height, width, d_model // 2)
        pos_emb_w = self.pos_enc_w.unsqueeze(0).repeat(height, 1, 1)  # Shape: (height, width, d_model // 2)
        pos_emb = torch.cat([pos_emb_h, pos_emb_w], dim=-1)  # Shape: (height, width, d_model)

        # Add the positional encoding to the input tensor
        x = x + pos_emb.permute(2, 0, 1).unsqueeze(0)  # Shape: (batch_size, channels, height, width, d_model)

        # Reshape the input tensor for processing by the TransformerEncoder
        x = x.permute(0, 3, 2, 1).reshape(batch_size * width, height, channels)

        # Apply the TransformerEncoder layer to each spatial location separately
        x = self.transformer_encoder(x)

        # Reshape the output tensor back to the original shape
        x = x.reshape(batch_size, width, height, channels, -1).permute(0, 3, 2, 1, 4)
        
        return x.squeeze()
    

class SpatialTransformerEncoderP5(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=3):
        super(SpatialTransformerEncoderP5, self).__init__()

        # Define the positional encoding for the spatial dimensions
        h, w = 12, 20  # Height and width of the input images
        pos_enc_h = nn.Parameter(torch.randn(h, d_model // 2))
        pos_enc_w = nn.Parameter(torch.randn(w, d_model // 2))

        # Define the TransformerEncoder layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_layers)

        # Save the parameters
        self.pos_enc_h = pos_enc_h
        self.pos_enc_w = pos_enc_w

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Compute the positional encodings for the spatial dimensions
        pos_emb_h = self.pos_enc_h.unsqueeze(1).repeat(1, width, 1)  # Shape: (height, width, d_model // 2)
        pos_emb_w = self.pos_enc_w.unsqueeze(0).repeat(height, 1, 1)  # Shape: (height, width, d_model // 2)
        pos_emb = torch.cat([pos_emb_h, pos_emb_w], dim=-1)  # Shape: (height, width, d_model)

        # Add the positional encoding to the input tensor
        x = x + pos_emb.permute(2, 0, 1).unsqueeze(0)  # Shape: (batch_size, channels, height, width, d_model)

        # Reshape the input tensor for processing by the TransformerEncoder
        x = x.permute(0, 3, 2, 1).reshape(batch_size * width, height, channels)

        # Apply the TransformerEncoder layer to each spatial location separately
        x = self.transformer_encoder(x)

        # Reshape the output tensor back to the original shape
        x = x.reshape(batch_size, width, height, channels, -1).permute(0, 3, 2, 1, 4)
        
        return x.squeeze()

class SpatialTransformerEncoderP6(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=3):
        super(SpatialTransformerEncoderP6, self).__init__()

        # Define the positional encoding for the spatial dimensions
        h, w = 6, 10  # Height and width of the input images
        pos_enc_h = nn.Parameter(torch.randn(h, d_model // 2))
        pos_enc_w = nn.Parameter(torch.randn(w, d_model // 2))

        # Define the TransformerEncoder layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_layers)

        # Save the parameters
        self.pos_enc_h = pos_enc_h
        self.pos_enc_w = pos_enc_w

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Compute the positional encodings for the spatial dimensions
        pos_emb_h = self.pos_enc_h.unsqueeze(1).repeat(1, width, 1)  # Shape: (height, width, d_model // 2)
        pos_emb_w = self.pos_enc_w.unsqueeze(0).repeat(height, 1, 1)  # Shape: (height, width, d_model // 2)
        pos_emb = torch.cat([pos_emb_h, pos_emb_w], dim=-1)  # Shape: (height, width, d_model)

        # Add the positional encoding to the input tensor
        x = x + pos_emb.permute(2, 0, 1).unsqueeze(0)  # Shape: (batch_size, channels, height, width, d_model)

        # Reshape the input tensor for processing by the TransformerEncoder
        x = x.permute(0, 3, 2, 1).reshape(batch_size * width, height, channels)

        # Apply the TransformerEncoder layer to each spatial location separately
        x = self.transformer_encoder(x)

        # Reshape the output tensor back to the original shape
        x = x.reshape(batch_size, width, height, channels, -1).permute(0, 3, 2, 1, 4)
        
        return x.squeeze()

class SpatialTransformerEncoderP7(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=3):
        super(SpatialTransformerEncoderP7, self).__init__()

        # Define the positional encoding for the spatial dimensions
        h, w = 3, 5  # Height and width of the input images
        pos_enc_h = nn.Parameter(torch.randn(h, d_model // 2))
        pos_enc_w = nn.Parameter(torch.randn(w, d_model // 2))

        # Define the TransformerEncoder layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_layers)

        # Save the parameters
        self.pos_enc_h = pos_enc_h
        self.pos_enc_w = pos_enc_w

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Compute the positional encodings for the spatial dimensions
        pos_emb_h = self.pos_enc_h.unsqueeze(1).repeat(1, width, 1)  # Shape: (height, width, d_model // 2)
        pos_emb_w = self.pos_enc_w.unsqueeze(0).repeat(height, 1, 1)  # Shape: (height, width, d_model // 2)
        pos_emb = torch.cat([pos_emb_h, pos_emb_w], dim=-1)  # Shape: (height, width, d_model)

        # Add the positional encoding to the input tensor
        x = x + pos_emb.permute(2, 0, 1).unsqueeze(0)  # Shape: (batch_size, channels, height, width, d_model)

        # Reshape the input tensor for processing by the TransformerEncoder
        x = x.permute(0, 3, 2, 1).reshape(batch_size * width, height, channels)

        # Apply the TransformerEncoder layer to each spatial location separately
        x = self.transformer_encoder(x)

        # Reshape the output tensor back to the original shape
        x = x.reshape(batch_size, width, height, channels, -1).permute(0, 3, 2, 1, 4)
        
        return x.squeeze()