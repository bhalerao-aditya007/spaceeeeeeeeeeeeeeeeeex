import torch

checkpoint = torch.load('perception/checkpoints/best.pt',
                        map_location='cpu',
                        weights_only=False)

print("Epoch:", checkpoint['epoch'])
print("Rot error:", checkpoint['rot_err_deg'])
print("Trans error:", checkpoint['trans_err_m'])
print("\nConfig:")
print(checkpoint['cfg'])