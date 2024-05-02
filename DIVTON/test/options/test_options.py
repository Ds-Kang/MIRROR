from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--warp_checkpoint', type=str, default='./checkpoints/PFAFN_warp_epoch_101_phif_pl_skip.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--gen_checkpoint', type=str, default='./checkpoints/PFAFN_gen_epoch_101_phif_pl_skip.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--parsing', type=str, default="pascal", help='label for parsing network')
        self.parser.add_argument('--parsing_checkpoint', type=str, default="checkpoints/exp-schp-201908270938-pascal-person-part.pth", help='load the pretrained model from the specified location')
        self.parser.add_argument('--skip', type=bool, default=True)
        
        self.isTrain = False
