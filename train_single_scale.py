from config import Config
import os
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import interpolate
from loguru import logger
from tqdm import tqdm
import numpy as np
import wandb

from draw_concat import draw_concat
from generate_noise import generate_spatial_noise
from minecraft.level_utils import one_hot_to_blockdata_level, save_level_to_world, clear_empty_world
from minecraft.level_renderer import render_minecraft
from models import calc_gradient_penalty, save_networks
from utils import interpolate3D


def update_noise_amplitude(z_prev, real, opt):
    """ Update the amplitude of the noise for the current scale according to the previous noise map. """
    RMSE = torch.sqrt(F.mse_loss(real, z_prev))
    return opt.noise_update * RMSE


def train_single_scale(D, G, reals, generators, noise_maps, input_from_prev_scale, noise_amplitudes, opt):
    """ Train one scale. D and G are the current discriminator and generator, reals are the scaled versions of the
    original level, generators and noise_maps contain information from previous scales and will receive information in
    this scale, input_from_previous_scale holds the noise map and images from the previous scale, noise_amplitudes hold
    the amplitudes for the noise in all the scales. opt is a namespace that holds all necessary parameters. """
    current_scale = len(generators)

    if not opt.server_train:
        clear_empty_world(opt.output_dir, 'Curr_Empty_World')  # reset tmp world

    if opt.use_multiple_inputs:
        real_group = []
        nzx_group = []
        nzy_group = []
        nz_group = []
        for scale_group in reals:
            real_group.append(scale_group[current_scale])
            nzx_group.append(scale_group[current_scale].shape[2])
            nzy_group.append(scale_group[current_scale].shape[3])
            nz_group.append(
                (scale_group[current_scale].shape[2], scale_group[current_scale].shape[3]))

        curr_noises = [0 for _ in range(len(real_group))]
        curr_prevs = [0 for _ in range(len(real_group))]
        curr_z_prevs = [0 for _ in range(len(real_group))]

    else:
        real = reals[current_scale]
        nz = real.shape[2:]

    # As kernel size is always 3 currently, padsize goes up by one per layer
    padsize = int(1 * opt.num_layer)

    if not opt.pad_with_noise:
        # pad_noise = nn.ConstantPad3d(padsize, 0)
        # pad_image = nn.ConstantPad3d(padsize, 0)
        pad_noise = nn.ReplicationPad3d(padsize)
        pad_image = nn.ReplicationPad3d(padsize)

    else:
        pad_noise = nn.ReplicationPad3d(padsize)
        pad_image = nn.ReplicationPad3d(padsize)

    # setup optimizer
    optimizerD = optim.Adam(D.parameters(), lr=opt.lr_d,
                            betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=opt.lr_g,
                            betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizerD, milestones=[1600, 2500], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizerG, milestones=[1600, 2500], gamma=opt.gamma)

    if current_scale == 0:  # Generate new noise
        if opt.use_multiple_inputs:
            z_opt_group = []
            for nzx, nzy in zip(nzx_group, nzy_group):
                z_opt = generate_spatial_noise(
                    [1, opt.nc_current, nzx, nzy], device=opt.device)
                z_opt = pad_noise(z_opt)
                z_opt_group.append(z_opt)
        else:
            z_opt = generate_spatial_noise(
                (1, opt.nc_current) + nz, device=opt.device)
            z_opt = pad_noise(z_opt)
    else:  # Add noise to previous output
        if opt.use_multiple_inputs:
            z_opt_group = []
            for nzx, nzy in zip(nzx_group, nzy_group):
                z_opt = torch.zeros(
                    [1, opt.nc_current, nzx, nzy]).to(opt.device)
                z_opt = pad_noise(z_opt)
                z_opt_group.append(z_opt)
        else:
            z_opt = torch.zeros((1, opt.nc_current) + nz).to(opt.device)
            z_opt = pad_noise(z_opt)

    logger.info("Training at scale {}", current_scale)
    grad_d_real = []
    grad_d_fake = []
    grad_g = []
    for p in D.parameters():
        grad_d_real.append(torch.zeros(p.shape).to(opt.device))
        grad_d_fake.append(torch.zeros(p.shape).to(opt.device))

    for p in G.parameters():
        grad_g.append(torch.zeros(p.shape).to(opt.device))

    for epoch in tqdm(range(opt.niter)):
        step = current_scale * opt.niter + epoch
        if opt.use_multiple_inputs:
            group_steps = len(real_group)
            noise_group = []
            for nzx, nzy in zip(nzx_group, nzy_group):
                noise_ = generate_spatial_noise(
                    [1, opt.nc_current, nzx, nzy], device=opt.device)
                noise_ = pad_noise(noise_)
                noise_group.append(noise_)
        else:
            group_steps = 1
            noise_ = generate_spatial_noise(
                (1, opt.nc_current) + nz, device=opt.device)
            noise_ = pad_noise(noise_)

        for curr_inp in range(group_steps):
            if opt.use_multiple_inputs:
                real = real_group[curr_inp]
                nz = nz_group[curr_inp]
                z_opt = z_opt_group[curr_inp]
                noise_ = noise_group[curr_inp]
                prev_scale_results = input_from_prev_scale[curr_inp]
                opt.curr_inp = curr_inp
            else:
                prev_scale_results = input_from_prev_scale

            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            for j in range(opt.Dsteps):
                # train with real
                D.zero_grad()

                output = D(real).to(opt.device)

                errD_real = -output.mean()

                errD_real.backward(retain_graph=True)

                # grads_after = []
                # cos_sim = []
                # for i, p in enumerate(D.parameters()):
                #     grads_after.append(p.grad)
                #     cos_sim.append(nn.CosineSimilarity(-1)(grad_d_real[i], p.grad).mean().item())

                # diff_d_real = np.mean(cos_sim)

                # grad_d_real = grads_after

                # train with fake
                if (j == 0) & (epoch == 0):
                    if current_scale == 0:  # If we are in the lowest scale, noise is generated from scratch
                        prev = torch.zeros(
                            (1, opt.nc_current) + nz).to(opt.device)
                        prev_scale_results = prev
                        prev = pad_image(prev)
                        z_prev = torch.zeros(
                            (1, opt.nc_current) + nz).to(opt.device)
                        z_prev = pad_noise(z_prev)
                        opt.noise_amp = 1
                    else:  # First step in NOT the lowest scale
                        # We need to adapt our inputs from the previous scale and add noise to it
                        prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, prev_scale_results,
                                           "rand", pad_noise, pad_image, opt)

                        prev = interpolate3D(
                            prev, real.shape[-3:], mode="bilinear", align_corners=True)

                        prev = pad_image(prev)
                        z_prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, prev_scale_results,
                                             "rec", pad_noise, pad_image, opt)
                        z_prev = interpolate3D(
                            z_prev, real.shape[-3:], mode="bilinear", align_corners=True)
                        opt.noise_amp = update_noise_amplitude(
                            z_prev, real, opt)
                        z_prev = pad_image(z_prev)
                else:  # Any other step
                    if opt.use_multiple_inputs:
                        z_prev = curr_z_prevs[curr_inp]

                    prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, prev_scale_results,
                                       "rand", pad_noise, pad_image, opt)

                    prev = interpolate3D(
                        prev, real.shape[-3:], mode="bilinear", align_corners=False)

                    prev = pad_image(prev)

                # After creating our correct noise input, we feed it to the generator:
                noise = opt.noise_amp * noise_ + prev
                fake = G(noise.detach(), prev)

                # Then run the result through the discriminator
                output = D(fake.detach())
                errD_fake = output.mean()

                # Backpropagation
                errD_fake.backward(retain_graph=False)

                # Gradient Penalty
                gradient_penalty = calc_gradient_penalty(
                    D, real, fake, opt.lambda_grad, opt.device)
                gradient_penalty.backward(retain_graph=False)

                # grads_after = []
                # cos_sim = []
                # for i, p in enumerate(D.parameters()):
                #     grads_after.append(p.grad)
                #     cos_sim.append(nn.CosineSimilarity(-1)(grad_d_fake[i], p.grad).mean().item())

                # diff_d_fake = np.mean(cos_sim)

                # grad_d_fake = grads_after

                # Logging:
                if step % 10 == 0:
                    wandb.log({f"D(G(z))@{current_scale}": errD_fake.item(),
                               f"D(x)@{current_scale}": -errD_real.item(),
                               f"gradient_penalty@{current_scale}": gradient_penalty.item(),
                               #    f"D_real_grad@{current_scale}": diff_d_real,
                               #    f"D_fake_grad@{current_scale}": diff_d_fake,
                               },
                              step=step, sync=False)
                optimizerD.step()

                if opt.use_multiple_inputs:
                    z_opt_group[curr_inp] = z_opt
                    input_from_prev_scale[curr_inp] = prev_scale_results
                    curr_noises[curr_inp] = noise
                    curr_prevs[curr_inp] = prev
                    curr_z_prevs[curr_inp] = z_prev

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################

            for j in range(opt.Gsteps):
                G.zero_grad()
                fake = G(noise.detach(), prev.detach(), temperature=1)
                output = D(fake)

                errG = -output.mean()
                errG.backward(retain_graph=False)

                # grads_after = []
                # cos_sim = []
                # for i, p in enumerate(G.parameters()):
                #     grads_after.append(p.grad)
                #     cos_sim.append(nn.CosineSimilarity(-1)(grad_g[i], p.grad).mean().item())

                # diff_g = np.mean(cos_sim)

                # grad_g = grads_after

                if opt.alpha != 0:  # i. e. we are trying to find an exact recreation of our input in the lat space
                    Z_opt = opt.noise_amp * z_opt + z_prev
                    G_rec = G(Z_opt.detach(), z_prev, temperature=1)
                    rec_loss = opt.alpha * F.mse_loss(G_rec, real)
                    # TODO: Check for unexpected argument retain_graph=True
                    rec_loss.backward(retain_graph=False)
                    rec_loss = rec_loss.detach()
                else:  # We are not trying to find an exact recreation
                    rec_loss = torch.zeros([])
                    Z_opt = z_opt

                optimizerG.step()

        # More Logging:
        if step % 10 == 0:
            wandb.log({f"noise_amplitude@{current_scale}": opt.noise_amp,
                       f"rec_loss@{current_scale}": rec_loss.item(),
                       #    f"G_grad@{current_scale}": diff_g
                       },
                      step=step, sync=False, commit=True)

        # Rendering and logging images of levels
        if epoch % 500 == 0 or epoch == (opt.niter - 1):
            token_list = opt.token_list

            to_level = one_hot_to_blockdata_level

            if opt.render:
                if not opt.server_train:
                    try:
                        subprocess.call(["wine", '--version'])
                        real_scaled = to_level(
                            real.detach(), token_list, opt.block2repr, opt.repr_type)

                        # Minecraft World
                        worldname = 'Curr_Empty_World'
                        clear_empty_world(opt.output_dir, worldname)  # reset tmp world
                        to_render = [real_scaled, to_level(fake.detach(), token_list, opt.block2repr, opt.repr_type),
                                     to_level(G(Z_opt.detach(), z_prev), token_list, opt.block2repr, opt.repr_type)]
                        render_names = [
                            f"real@{current_scale}", f"G(z)@{current_scale}", f"G(z_opt)@{current_scale}"]
                        obj_pth = os.path.join(opt.out_, f"objects/{current_scale}")
                        os.makedirs(obj_pth, exist_ok=True)
                        if opt.repr_type == "neighbert" or opt.repr_type == "one-hot-neighbors":
                            token_list = opt.token_list_translation
                        for n, level in enumerate(to_render):
                            pos = n * (level.shape[0] + 5)
                            save_level_to_world(
                                opt.output_dir, worldname, (pos, 0, 0), level, token_list, opt.props)
                            curr_coords = [[pos, pos + real_scaled.shape[0]],
                                           [0, real_scaled.shape[1]],
                                           [0, real_scaled.shape[2]]]
                            render_pth = render_minecraft(
                                worldname, curr_coords, obj_pth, render_names[n])
                            rendered_images = render_world(render_pth, opt)
                            wandb.log({render_names[n]: wandb.Image(rendered_images)}, commit=False)
                    except OSError:
                        pass
                else:
                    # Server render!
                    t_level = to_level(G(Z_opt.detach(), z_prev), token_list, opt.block2repr, opt.repr_type)
                    render_to_server(t_level, opt)

            # Learning Rate scheduler step
            schedulerD.step()
            schedulerG.step()

    # Save networks

    if opt.use_multiple_inputs:
        z_opt = z_opt_group

    torch.save(z_opt, "%s/z_opt.pth" % opt.outf)
    save_networks(G, D, z_opt, opt)
    wandb.save(opt.outf)
    return z_opt, input_from_prev_scale, G


def render_to_server(curr_sample, ns_opt):
    # We're posting this directly on the server!
    ns_opt.mc_server.postToChat("Rendering sample...")

    # curr_sample = to_level(fake.detach(), token_list, opt.block2repr, opt.repr_type)
    for i in range(curr_sample.shape[0]):
        for j in range(curr_sample.shape[1]):
            for k in range(curr_sample.shape[2]):
                ns_opt.mc_server.setBlock(ns_opt.server_render_pos[0] + i,
                                          ns_opt.server_render_pos[1] + j,
                                          ns_opt.server_render_pos[2] + k,
                                          ns_opt.token_list[curr_sample[i, j, k].item()])

    ns_opt.mc_server.postToChat("Sample Rendered!")


def render_world(render_path: str, opt: Config, num_viewpoints: int = 20):
    from pytorch3d.io import load_objs_as_meshes
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        PointLights,
        Materials,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        HardFlatShader,
    )
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image
    import math
    mesh = load_objs_as_meshes([render_path], device=opt.device)
    meshes = mesh.extend(num_viewpoints)
    lights = PointLights(device=opt.device, location=[[0.0, 0.0, -3.0]])

    elev = torch.linspace(0, 180, num_viewpoints)
    azim = torch.linspace(-180, 180, num_viewpoints)

    R, T = look_at_view_transform(dist=3, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=opt.device, R=R, T=T)
    materials = Materials(
        device=opt.device,
        shininess=0.0
    )
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(
            device=opt.device,
            cameras=cameras,
            lights=lights
        )
    )

    # Move the light back in front of the cow which is facing the -z direction.
    lights.location = torch.tensor([[1.0, 1.0, -3.0]], device=opt.device)
    images = renderer(meshes, cameras=cameras,
                      lights=lights, materials=materials)
    image_grid = to_pil_image(make_grid(torch.permute(images[..., :3], dims=[0, 3, 1, 2]), nrow=math.floor(math.sqrt(num_viewpoints))))
    return image_grid
