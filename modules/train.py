import sys
import os

# import modules
package = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FrangiNet/'))
sys.path.append(package)

from train import main_frangi_train

if len(sys.argv) > 1:
    config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FrangiNet/')) + sys.argv[1] + '.json'
else:
    config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FrangiNet/')) + "config_train_model.json"

main_frangi_train(config_file)
