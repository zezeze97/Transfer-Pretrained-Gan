import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor 
import torch

image_feature_extractor = models.resnet18(pretrained=True)
nodes = get_graph_node_names(image_feature_extractor)
return_nodes = {'layer4.1.relu_1': 'layer4',
                'layer3.1.relu_1': 'layer3',
                'layer2.1.relu_1': 'layer2',
                'layer1.1.relu_1': 'layer1' }
image_feature_extractor = create_feature_extractor(image_feature_extractor, return_nodes=return_nodes)
