from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from opts import opts
from detectors.detector_factory import detector_factory

from ptflops import get_model_complexity_info


def get_flops(
    opt,
    input_size=(3, 224, 224),
):
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    model = detector.model
    macs, params = get_model_complexity_info(
        model, input_size, as_strings=True, print_per_layer_stat=False, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))


if __name__ == "__main__":
    opt = opts().init()
    get_flops(opt)
