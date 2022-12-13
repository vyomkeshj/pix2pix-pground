from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """
    def __init__(self):
        super().__init__()
        self.isTrain = False

    def initialize(self, parser):

        parser = BaseOptions.initialize(self, parser)  # define shared options

        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')

        return parser
