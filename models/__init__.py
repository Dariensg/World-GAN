# Contains code based on https://github.com/tamarott/SinGAN
import os
import torch


from .generator import Level_GeneratorConcatSkip2CleanAdd
from .discriminator import Level_WDiscriminator


def weights_init(m):
    """ Init weights for Conv and Norm Layers. """
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("Conv3d") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("Norm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_models(opt, use_softmax=True):
    """ Initialize Generator and Discriminator. """
    # generator initialization:
    G = Level_GeneratorConcatSkip2CleanAdd(opt, use_softmax).to(opt.device)
    G.apply(weights_init)
    if opt.netG != "":
        G.load_state_dict(torch.load(opt.netG))
    print(G)

    # discriminator initialization:
    D1 = Level_WDiscriminator(opt).to(opt.device)
    D1.apply(weights_init)
    if opt.netD != "":
        D1.load_state_dict(torch.load(opt.netD))
    print(D1)

    D2 = Level_WDiscriminator(opt).to(opt.device)
    D2.apply(weights_init)
    if opt.netD != "":
        D2.load_state_dict(torch.load(opt.netD))
    print(D2)

    return D1, D2, G


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def save_networks(G, D1, D2, z_opt, opt):
    torch.save(G.state_dict(), "%s/G.pth" % (opt.outf))
    torch.save(D1.state_dict(), "%s/D1.pth" % (opt.outf))
    torch.save(D2.state_dict(), "%s/D2.pth" % (opt.outf))
    torch.save(z_opt, "%s/z_opt.pth" % (opt.outf))


def restore_weights(D1_curr, D2_curr, G_curr, scale_num, opt):
    G_state_dict = torch.load("%s/%d/G.pth" % (opt.out_, scale_num - 1))
    D1_state_dict = torch.load("%s/%d/D1.pth" % (opt.out_, scale_num - 1))
    D2_state_dict = torch.load("%s/%d/D2.pth" % (opt.out_, scale_num - 1))

    G_head_conv_weight = G_state_dict["head.conv.weight"]
    G_state_dict["head.conv.weight"] = G_curr.head.conv.weight
    G_tail_weight = G_state_dict["tail.0.weight"]
    G_state_dict["tail.0.weight"] = G_curr.tail[0].weight
    G_tail_bias = G_state_dict["tail.0.bias"]
    G_state_dict["tail.0.bias"] = G_curr.tail[0].bias
    D1_head_conv_weight = D1_state_dict["head.conv.weight"]
    D1_state_dict["head.conv.weight"] = D1_curr.head.conv.weight
    D2_head_conv_weight = D2_state_dict["head.conv.weight"]
    D2_state_dict["head.conv.weight"] = D2_curr.head.conv.weight


    G_state_dict["head.conv.weight"] = (G_state_dict["head.conv.weight"].detach().requires_grad_())
    G_state_dict["tail.0.weight"] = (G_state_dict["tail.0.weight"].detach().requires_grad_())
    G_state_dict["tail.0.bias"] = G_state_dict["tail.0.bias"].detach().requires_grad_()
    D1_state_dict["head.conv.weight"] = (D1_state_dict["head.conv.weight"].detach().requires_grad_())
    D2_state_dict["head.conv.weight"] = (D2_state_dict["head.conv.weight"].detach().requires_grad_())

    G_curr.load_state_dict(G_state_dict)
    D1_curr.load_state_dict(D1_state_dict)
    D2_curr.load_state_dict(D2_state_dict)

    G_curr.head.conv.weight = torch.nn.Parameter(G_curr.head.conv.weight.detach().requires_grad_())
    G_curr.tail[0].weight = torch.nn.Parameter(G_curr.tail[0].weight.detach().requires_grad_())
    G_curr.tail[0].bias = torch.nn.Parameter(G_curr.tail[0].bias.detach().requires_grad_())
    D1_curr.head.conv.weight = torch.nn.Parameter(D1_curr.head.conv.weight.detach().requires_grad_())
    D2_curr.head.conv.weight = torch.nn.Parameter(D2_curr.head.conv.weight.detach().requires_grad_())

    return D1_curr, D2_curr, G_curr


def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model


def load_trained_pyramid(opt):
    dir = opt.out_
    if(os.path.exists(dir)):
        reals = torch.load('%s/reals.pth' % dir)
        discriminator1_reals = torch.load('%s/discriminator1_reals.pth' % dir)
        discriminator2_reals = torch.load('%s/discriminator2_reals.pth' % dir)
        Gs = torch.load('%s/generators.pth' % dir)
        Zs = torch.load('%s/noise_maps.pth' % dir)
        NoiseAmp = torch.load('%s/noise_amplitudes.pth' % dir)

    else:
        print('no appropriate trained model exists, please train first')
    return Gs,Zs,reals,NoiseAmp,discriminator1_reals,discriminator2_reals