import testmemsam


def parse_args(argv=None):
    args = testmemsam.parse_args(argv, defaults={
        'dino_use_lora': True,
        'enable_phase_memory': True,
        'semi': False,
    })
    if args.task in ('EchoNet_Video', 'EchoDynamic', 'EchoDynamic_Video'):
        args.semi = True
    return args


def main():
    testmemsam.main(args=parse_args())


if __name__ == '__main__':
    main()
