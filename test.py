import sys
from FrangiNet.test import main_frangi_test

if len(sys.argv) > 1:
    config_file = 'FrangiNet/' + sys.argv[1] + '.json'
else:
    config_file = "FrangiNet/config_test_model.json"

main_frangi_test(config_file)
