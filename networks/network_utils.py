from networks.U_Net import U_Net
from networks.Mid_U_Net import Mid_U_Net

def set_model(model_name, in_channels=2, out_channels=3, out_layers=1):
    if model_name == 'U_Net':
        model  = U_Net(in_channels=in_channels, out_channels=out_channels, out_layers=out_layers)
    elif model_name == 'Mid_U_Net':
        model  = Mid_U_Net(in_channels=in_channels, out_channels=out_channels, out_layers=out_layers)
    elif model_name == 'Big_U_Net':
        model = None

    return model