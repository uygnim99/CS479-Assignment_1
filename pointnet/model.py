import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        if x.size(dim=1) != self.k:
            x = torch.transpose(x, 1, 2)
        # print(f"x1: {x.shape}, k: {self.k}, device: {x.device}, type: {type(x)}")
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)

        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """

    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.

        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU()
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """

        # print(f"stn3 output: {self.stn3(pointcloud).shape}")
        # print(f"pointcloud shape: {pointcloud.shape}")
        # TODO : Implement forward function.
        if self.input_transform:
            transform = self.stn3(pointcloud)
            output = torch.matmul(transform.unsqueeze(
                dim=1), pointcloud.unsqueeze(dim=3))
        else:
            output = pointcloud
        # print(f"output.shape before mlp1: {output.shape}")
        # [B, N, 3, 1] -> [B, 3, N]
        output = torch.transpose(output.squeeze(dim=-1), 1, 2)
        output = self.mlp1(output)  # [B, 3, N] -> [B, 64, N]
        # print(f"output.shape after mlp1: {output.shape}")
        if self.feature_transform:
            transform = self.stn64(output)   # transform matrix [B, 64, 64]
            output = torch.matmul(transform.unsqueeze(
                dim=1), output.unsqueeze(dim=3))  # matrix multiplication [B, 1, 64, 64] * [B, N, 64, 1]
        output = torch.transpose(output.squeeze(dim=-1), 1, 2)
        output = self.mlp2(output)  # [B, N, 1024]
        output = torch.max(output, 2, keepdim=False)  # [B, 1024]
        return output


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes

        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)

        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.mlp = nn.Sequential(
            nn.Conv1d(1024, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3), nn.Conv1d(256, num_classes, 1)
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.
        values, indices = self.pointnet_feat(pointcloud)
        # print(f"PointNetCls output")
        # print(f"values.shape: {values.shape}")
        # print(f"indices.shape: {indices.shape}")
        output = self.mlp(values.unsqueeze(dim=-1))
        # print(f"output.shape: {output.shape}")
        return F.log_softmax(output.squeeze(), dim=1)  # [B, num_classes]


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.

        self.stn3 = STNKd(k=3)
        self.stn64 = STNKd(k=64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.mlp4 = nn.Sequential(
            nn.Conv1d(128, m, 1), nn.BatchNorm1d(m), nn.ReLU()
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        transform = self.stn3(pointcloud)  # transform matrix [B, 3, 3]
        output = torch.matmul(transform.unsqueeze(
            dim=1), pointcloud.unsqueeze(dim=3))  # matrix multiplication [B, N, 3, 3] * [B, N, 3, 1]
        # [B, N, 3, 1] -> [B, 3, N]
        output = torch.transpose(output.squeeze(dim=-1), 1, 2)
        output = self.mlp1(output)  # [B, 3, N] -> [B, 64, N]
        transform = self.stn64(output)  # [B, N, 64, 64]
        # print(f"test: {transform.unsqueeze(dim=1).shape}")
        # print(f"test: {output.unsqueeze(dim=3).shape}")
        output = torch.transpose(output, 1, 2)
        output = torch.matmul(transform.unsqueeze(
            dim=1), output.unsqueeze(dim=3))  # matrix multiplication [B, 1, 64, 64] * [B, N, 64, 1]
        # [B, N, 64, 1] -> [B, 64, N]
        local_feature = torch.transpose(output.squeeze(dim=-1), 1, 2)
        output = self.mlp2(local_feature)  # [B, N, 1024]
        global_feature, indices = torch.max(
            output, 2, keepdim=False)  # [B, 1024]
        global_feature = global_feature.unsqueeze(
            dim=-1).expand(-1, -1, local_feature.size(2))  # [B, 1024] -> [B, 1024, N]
        feature = torch.cat((local_feature, global_feature), 1)
        point_feature = self.mlp3(feature)  # [B, 128, N]
        scores = self.mlp4(point_feature)  # [B, m, N]

        return scores


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat(False, False)

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.
        self.mlp_decoder = nn.Sequential(
            nn.Conv1d(1024, num_points/4, 1),
            nn.BatchNorm1d(num_points/4), nn.ReLU(),
            nn.Conv1d(num_points/4, num_points/2, 1),
            nn.BatchNorm1d(num_points/2), nn.ReLU(),
            nn.Conv1d(num_points/2, num_points, 1),
            nn.Dropout(), nn.BatchNorm1d(num_points), nn.ReLU(),
            nn.Conv1d(num_points, num_points*3, 1),
            nn.BatchNorm1d(num_points*3), nn.ReLU(),
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        values, indices = self.pointnet_feat(pointcloud)  # [B, 1024]
        output = self.mlp_decoder(values.unsqueeze(dim=-1))  # [B, N*3, 1]
        return torch.reshape(output.squeeze(), pointcloud.shape)  # [B, N, 3]


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
