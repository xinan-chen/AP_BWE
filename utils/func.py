import torch

def fAW(x):
    return torch.abs(x-2*torch.pi*torch.round(x/(2*torch.pi)))

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    # r_losses = []
    # g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr).clamp(min=0.0))
        g_loss = torch.mean((1+dg).clamp(min=0.0))
        loss += (r_loss + g_loss)
        # r_losses.append(r_loss.item())
        # g_losses.append(g_loss.item())

    return loss


def generator_loss(disc_outputs):
    loss = 0
    # gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg).clamp(min=0.0))
        # gen_losses.append(l)
        loss += l
        
    return loss