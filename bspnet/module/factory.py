from .bsp_2d.model import BSP2dModel


def factory(phase=0):
    model = BSP2dModel(
        base_channels=32,
        num_planes=4096,
        num_primitives=256,
        resolution=64,
        phase=phase,
    )

    return model
