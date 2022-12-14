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

        parser.set_defaults(model='test')
        parser.add_argument('--phase', type=str, default='test', help='train, test, etc')
        parser.add_argument('--output_dir', type=str, default='./result', help='Where to store the inference result')

        return parser
