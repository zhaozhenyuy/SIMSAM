from models.segment_anything_memsam.build_memsam import memsam_model_registry


def get_model(modelname="SharedGroundedMemSAM", args=None, opt=None):
    if modelname != "SharedGroundedMemSAM":
        raise ValueError(f"This release only provides SharedGroundedMemSAM, got {modelname!r}.")
    return memsam_model_registry["shared_grounded"](
        args=args,
        checkpoint=args.sam_ckpt,
    )
