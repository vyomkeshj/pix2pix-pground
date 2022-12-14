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
        parser.add_argument('--output_dir', type=str, default='./variant_2_result', help='Where to store the inference variant_2_result')
        parser.add_argument('--rgb_dir', type=str, default='./select_rgb', help='Where to store the inference variant_2_result')
        parser.add_argument('--seg_dir', type=str, default='./select_masks', help='Where to store the inference variant_2_result')

        return parser
