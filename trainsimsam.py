import trainmemsam


def parse_args(argv=None):
    return trainmemsam.parse_args(argv, defaults={
        'modelname': 'SharedGroundedMemSAM',
        'enable_memory': True,
        'semi': True,
        'disable_point_prompt': True,
        'dino_use_lora': True,
        'enable_phase_memory': True,
    })


def main():
    trainmemsam.main(args=parse_args())


if __name__ == '__main__':
    main()
