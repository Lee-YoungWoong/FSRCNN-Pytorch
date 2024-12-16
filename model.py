import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init                
                
class FSRCNN(nn.Module):
    def __init__(self, img_format, scale_factor) -> None:
        super(FSRCNN, self).__init__()

        if img_format == 'RGB' or img_format == 'YCbCr': 
            input_channels = 3
        elif img_format == 'Y':
            input_channels = 1
        else:
            raise ValueError('Image format not supported')
        
        #Paper Parameters
        d = 56
        s = 12

        # Feature extraction Layer
        self.feature_extract = nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, padding=2)
        nn.init.kaiming_normal_(self.feature_extract.weight)
        nn.init.zeros_(self.feature_extract.bias)

        self.activation_1st = nn.PReLU(num_parameters=d)

        # Shrink Layer
        self.shrink = nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1)
        nn.init.kaiming_normal_(self.shrink.weight)
        nn.init.zeros_(self.shrink.bias)

        self.activation_2nd = nn.PReLU(num_parameters=s)
        
        # Mapping Layer(m=4)
        self.map_1st = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_1st.weight)
        nn.init.zeros_(self.map_1st.bias)

        self.map_2nd = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_2nd.weight)
        nn.init.zeros_(self.map_2nd.bias)

        self.map_3rd = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_3rd.weight)
        nn.init.zeros_(self.map_3rd.bias)

        self.map_4th = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_4th.weight)
        nn.init.zeros_(self.map_4th.bias)

        self.activation_3rd = nn.PReLU(num_parameters=s)
        
        # Expand Layer
        self.expand = nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1)
        nn.init.kaiming_normal_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

        self.activation_4th = nn.PReLU(num_parameters=d)

        # Deconvolution Layer(Up-sampling)
        self.deconv = nn.ConvTranspose2d(in_channels=d, out_channels=input_channels, kernel_size=9, stride=scale_factor, padding=3, output_padding=scale_factor-1)
        nn.init.normal_(self.deconv.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias)

    def forward(self, X_in):
        # Feature extraction Layer
        X = self.feature_extract(X_in)
        X = self.activation_1st(X)

        # Shrink Layer
        X = self.shrink(X)
        X = self.activation_2nd(X)

        # Mapping Layer(m=4)
        X = self.map_1st(X)
        X = self.map_2nd(X)
        X = self.map_3rd(X)
        X = self.map_4th(X)
        X = self.activation_3rd(X)
        
        # Expand Layer
        X = self.expand(X)
        X = self.activation_4th(X)

        # Deconvolution Layer(Up-sampling)
        X = self.deconv(X)
        X_out = torch.clip(X, 0.0, 1.0)     
           
        return X_out

        

