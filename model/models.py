from .linknet import LinkNet34, LinkNet34MTL
from .stack_module import StackHourglassNetMTL_DGCNv4
from .SPIN import spin
from .unet import unet

StackHourglassNetMTL =  None

MODELS = {"LinkNet34MTL": LinkNet34MTL, "StackHourglassNetMTL": StackHourglassNetMTL_DGCNv4, "SPIN": spin, "UNet": unet}

MODELS_REFINE = {"LinkNet34": LinkNet34}
