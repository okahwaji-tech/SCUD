
from omegaconf import OmegaConf

from scud.dit_text_sm import SMSCUD
from scud.unet import KingmaUNet
from scud.protein_convnet import ByteNetLMTimeNew
from scud.dit_text import SCUD

image_nn_name_dict = {
    "KingmaUNet":KingmaUNet,
}

protein_nn_name_dict = {
    "ConvNew": ByteNetLMTimeNew
}

text_nn_name_dict = {
    "DIT": SCUD,
    "SM_SCUD_TCR": SMSCUD,
    "SM_SCUD_PT": SMSCUD
}

def get_model_setup(cfg, tokenizer=None):
    schedule_conditioning = cfg.model.model in [
        "ScheduleCondition", "DiscreteScheduleCondition",
        "MaskingDiffusion",
        "SCUD", "SCUD_TCR", "SM_SCUD_TCR", "SM_SCUD_PT",
    ]
    nn_params = cfg.architecture.nn_params
    nn_params = (OmegaConf.to_container(nn_params, resolve=True)
            if nn_params is not None else {})
    if cfg.architecture.x0_model_class in image_nn_name_dict:
        nn_params = {
            "n_channel": 1 if cfg.data.data == 'MNIST' else 3, 
            "N": cfg.data.N + (cfg.model.model == 'MaskingDiffusion'),
            "n_T": cfg.model.n_T,
            "schedule_conditioning": schedule_conditioning,
            "s_dim": cfg.architecture.s_dim,
            **nn_params
        }
    
        return image_nn_name_dict[cfg.architecture.x0_model_class], nn_params
        
    elif cfg.architecture.x0_model_class in protein_nn_name_dict:
        nn_params = {
            "n_tokens": cfg.data.N + (cfg.model.model == 'MaskingDiffusion'),
            "schedule_conditioning": schedule_conditioning,
            **nn_params
        }
        return protein_nn_name_dict[cfg.architecture.x0_model_class], nn_params

    elif cfg.architecture.x0_model_class in text_nn_name_dict:
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for text models.")
        nn_params = {
            "n_tokens": len(tokenizer),
            **nn_params
        }
        return text_nn_name_dict[cfg.architecture.x0_model_class], nn_params
