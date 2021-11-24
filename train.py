import sys
from FrangiNet.train import main_frangi_train

if len(sys.argv) > 1:
    config_file = 'FrangiNet/' + sys.argv[1] + '.json'
else:
    config_file = "FrangiNet/config_train_model.json"

main_frangi_train(config_file)
