import torch
import numpy as np
import torch.nn.functional as F

from config import Config
from generate_samples import generate_samples
from minecraft.special_minecraft_downsampling import special_minecraft_downsampling
from minecraft.level_utils import clear_empty_world
from minecraft.level_utils import read_level as mc_read_level
from models import load_trained_pyramid
from utils import load_pkl
from edit_repr_space_experiment import adjust_token_list
import mcpi.minecraft as mc

sub_coord_dict = dict(
    ruins=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    simple_beach=[0.0, 0.5, 0.0, 1.0, 0.0, 1.0],
    desert=[0.25, 0.75, 0.0, 1.0, 0.25, 0.75],
    plains=[0.25, 0.75, 0.0, 1.0, 0.25, 0.75],
    swamp=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    vanilla_village=[0.33333, 0.66667, 0.0, 1.0, 0.33333, 0.66667],
    vanilla_mineshaft=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
)


class GenerateMCSamplesConfig(Config):
    out_: str = None  # folder containing generator files
    scale_v: float = 1.0  # vertical scale factor
    scale_h: float = 1.0  # horizontal scale factor
    scale_d: float = 1.0  # horizontal scale factor
    gen_start_scale: int = 0  # scale to start generating in
    num_samples: int = 10  # number of samples to be generated
    # save_tensors: bool = False  # save pytorch .pt tensors?
    not_cuda: bool = False  # disables cuda
    render_obj: bool = False  # if True make .obj files

    use_edited_b2v: bool = False  # Are we using the edited block2vec space or the normal one? (only with block2vec)

    def process_args(self):
        super().process_args()


if __name__ == '__main__':
    # config
    opt = GenerateMCSamplesConfig().parse_args()
    opt.sub_coords = sub_coord_dict[opt.input_area_name]
    opt.process_args()

    opt.game = 'minecraft'
    opt.ImgGen = None
    replace_tokens = None

    if not opt.server_train:
        clear_empty_world(opt.output_dir, opt.output_name)
    else:
        # connect to the minecraft server running RaspberryJuice
        opt.mc_server = mc.Minecraft.create()  # assumes default parameters on the server!

    downsample = special_minecraft_downsampling

    # Load Real
    real = mc_read_level(opt)
    opt.level_shape = real.shape[2:]

    # Load Generator
    generators, noise_maps, reals, noise_amplitudes = load_trained_pyramid(opt)

    # Get input shape for in_s
    real_down = downsample(1, [[opt.scale_v, opt.scale_h, opt.scale_d]], real, opt.token_list)
    real_down = real_down[0]
    in_s = torch.zeros_like(real_down, device=opt.device)
    if opt.use_edited_b2v:
        prefix = "b2v_edited"
        # old_block2repr = opt.block2repr
        opt.block2repr = load_pkl('edited_representations',
                                  prepath='/home/awiszus/Project/TOAD-GAN/input/minecraft/')
        # update token_list for rendering purposes
        # opt.token_list = adjust_token_list(opt.token_list)
    else:
        prefix = "arbitrary"
        # prefix = "all_scales"

    # Directory name
    s_dir_name = "%s_random_samples_v%.5f_h%.5f_d%.5f" % (
        prefix, opt.scale_v, opt.scale_h, opt.scale_d)

    generate_samples(generators, noise_maps, reals, noise_amplitudes, opt, in_s=in_s,
                     save_tensors=True, render_images=opt.render_obj,
                     scale_v=opt.scale_v, scale_h=opt.scale_h, scale_d=opt.scale_d,
                     save_dir=s_dir_name, num_samples=opt.num_samples, current_scale=0)

