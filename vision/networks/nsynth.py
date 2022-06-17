from networks.net_64 import EncoderVqResnet64, DecoderVqResnet64

 
class EncoderVq_resnet(EncoderVqResnet64):
    def __init__(self, dim_z, cfgs, flg_bn, flg_var_q):
        super(EncoderVq_resnet, self).__init__(1, dim_z, cfgs, flg_bn, flg_var_q)
        self.dataset = "NSynth"


class DecoderVq_resnet(DecoderVqResnet64):
    def __init__(self, dim_z, cfgs, flg_bn):
        super(DecoderVq_resnet, self).__init__(1, dim_z, cfgs, flg_bn)
        self.dataset = "NSynth"

