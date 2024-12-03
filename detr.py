import torch 
from torch import nn
from transformer import Transformer
from scipy.optimize import linear_sum_assignment


class DETR(nn.Module):
    def __init__(self, main_dim=64, ff_dim=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, num_classes=20, max_objects=100):
        super().__init__()
        self.backbone = nn.Sequential(
            self.downblock(3, 64),
            self.downblock(64, 128),
            self.downblock(128, 256),
            self.downblock(256, 512),
            self.downblock(512, 2048),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(2048, main_dim, 1),
            nn.BatchNorm2d(main_dim, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )
        self.transformer = Transformer(num_input_tokens=1, main_dim=main_dim, ff_dim=ff_dim, num_heads=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, max_tokens=512)
        self.transformer.fc = nn.Sequential()
        self.classifier = nn.Sequential(
            nn.Linear(main_dim, num_classes + 1),
            nn.Softmax(dim=2)
        )
        self.box_regressor = nn.Sequential(
            nn.Linear(main_dim, 4),
            nn.Sigmoid()
        )
        self.object_queries = nn.Parameter(torch.randn(max_objects, main_dim).type(torch.float), requires_grad=True)

    def downblock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )
    
    def forward(self, image):
        x = self.backbone(image)
        x = self.conv(x)
        batch_size, num_channels, h, w = x.size()
        x = x.reshape(batch_size, num_channels, h*w)
        x = x.permute(0,2,1)
        y = self.object_queries.repeat(batch_size, 1, 1)
        x = self.transformer(x, y)
        return self.box_regressor(x), self.classifier(x)
    

class DETRLoss(nn.Module):
    def __init__(self, iou_w, l1_w, num_classes) -> None:
        super().__init__()
        self.iou_w = iou_w
        self.l1_w = l1_w
        self.num_classes = num_classes

    def box_loss(self, bb1, bb2):
        bb1 = torch.unsqueeze(bb1, 1)
        bb2 = torch.unsqueeze(bb2, 0)
        xx1 = torch.maximum(bb2[..., 0], bb1[..., 0])
        yy1 = torch.maximum(bb2[..., 1], bb1[..., 1])
        xx2 = torch.minimum(bb2[..., 2], bb1[..., 2])
        yy2 = torch.minimum(bb2[..., 3], bb1[..., 3])
        w = torch.maximum(torch.tensor(0.), xx2 - xx1)
        h = torch.maximum(torch.tensor(0.), yy2 - yy1)
        wh = w * h
        iou = wh / ((bb2[..., 2] - bb2[..., 0]) * (bb2[..., 3] - bb2[..., 1])                                      
            + (bb1[..., 2] - bb1[..., 0]) * (bb1[..., 3] - bb1[..., 1]) - wh) 
        l1 = (bb1 - bb2).sum(-1).abs()          
        return(self.iou_w * (1-iou) + self.l1_w * l1)
    
    def optimial_loss(self, bb_true, bb_predicted, c_true, c_predicted):
        boxes_loss = self.box_loss(torch.index_select(bb_true, 0, torch.where(c_true!=self.num_classes)[0]), bb_predicted)
        probs_loss = torch.index_select(c_predicted, 1, c_true).transpose(1,0)
        valid_indexes, all_indexes = boxes_loss.shape
        cost_matrix = boxes_loss - probs_loss[:valid_indexes]
        indexes = list(linear_sum_assignment(cost_matrix.detach().cpu().numpy())[1])
        remain_indexes = set([i for i in range(len(bb_predicted))]).difference(set(indexes))
        indexes.extend(remain_indexes)
        indexes = torch.tensor(indexes).to(bb_true.device)
        loss1 = torch.zeros(size=(all_indexes,)).to(bb_true.device)
        loss1[:valid_indexes] = torch.diag(torch.index_select(boxes_loss, 1, indexes))
        loss2 = torch.diag(torch.index_select(probs_loss, 1, indexes))
        loss2 = (loss2 * torch.ones(size=(all_indexes,)).to(bb_true.device).index_fill(0, torch.arange(valid_indexes, all_indexes).to(bb_true.device), 1/10) + 1e-10).log()
        return (loss1 - loss2).sum()

    def forward(self, bb_true, bb_predicted, c_true, c_predicted):
        total_loss = 0
        for i in range(len(bb_true)):
            total_loss += self.optimial_loss(bb_true[i], bb_predicted[i], c_true[i], c_predicted[i])
        return total_loss / len(bb_true)