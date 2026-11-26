import torch

def load_model(path, device='cuda'):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except FileNotFoundError:
        return None

    if checkpoint.get('architecture') == 'DeepFillLite-two-stage-no-contextual-attention':
        from model.deepfill_lite import DeepFillLiteGenerator
        return DeepFillLiteGenerator(checkpoint).to(device).eval()

    gen_sd = checkpoint['G']
    if 'stage1.conv1.conv.weight' in gen_sd.keys():
        from model.networks import Generator
    else:
        from model.networks_tf import Generator

    gen = Generator(cnum_in=5, cnum=48, return_flow=False)
    gen = gen.to(device)
    gen.eval()

    gen.load_state_dict(gen_sd, strict=False)
    return gen
