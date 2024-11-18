
from datetime import datetime
time_format = r'%Y_%m_%dT%H_%M_%S'

ori_top1 = 0.8204

# ori_top5 = 0.9103

# accur 0.8349
mobilenet_path = './pre/mobilenet/best8204.pth'
# mobilenet_path = './pre/mobilenet/act_cifar.pth'
# mobilenet_path = './cifar.pth'
# mobilenet_path = './imagenet10.pth'
# mobilenet_path = './qmodel0.906.pth'


TB_DIR = 'runs/'

TIME_NOW = datetime.today().strftime(time_format)

# CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
# CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

# CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
