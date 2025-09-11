import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import convlstm, dconvlstm, convlstm_sac, dconvlstm_sac, convlstm_sac_temp


class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'convlstm': convlstm.ConvLSTM,
            'dconvlstm': dconvlstm.DConvLSTM,
            'convlstm_sac': convlstm_sac.ConvLSTM_SAC,
            'dconvlstm_sac': dconvlstm_sac.DConvLSTM_SAC,
            'convlstm_sac_temp': convlstm_sac_temp.ConvLSTM_SAC
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        #self.MSE_criterion = nn.MSELoss()
        self.MSE_criterion = nn.HuberLoss()

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt' + '-' + str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        # Forward pass: Get predictions and additional outputs
        outputs = self.network(frames_tensor, mask_tensor)
        if isinstance(outputs, tuple):  # Model returns multiple outputs
            next_frames, additional_info = outputs
        else:  # Backward compatibility if the model only returns next_frames
            next_frames = outputs
            additional_info = None

        # Calculate loss using next_frames
        loss = self.MSE_criterion(next_frames[:, :, :, :, 0], frames_tensor[:, 1:, :, :, 0])

        # Backward propagation
        loss.backward()
        self.optimizer.step()

        # Optionally process additional_info (e.g., for interpretability, debugging)
        if additional_info is not None:
            # Example: Log additional information
            pass

        return loss.detach().cpu().numpy()

    def valid(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)

        # Forward pass: Get predictions and additional outputs
        outputs = self.network(frames_tensor, mask_tensor)
        if isinstance(outputs, tuple):  # Model returns multiple outputs
            next_frames, additional_info = outputs
        else:
            next_frames = outputs
            additional_info = None

        # Calculate validation loss
        loss = self.MSE_criterion(next_frames[:, :, :, :, 0], frames_tensor[:, 1:, :, :, 0])

        # Optionally process additional_info
        if additional_info is not None:
            pass

        return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)

        # Forward pass: Get predictions and additional outputs
        outputs = self.network(frames_tensor, mask_tensor)
        if isinstance(outputs, tuple):  # Model returns multiple outputs
            next_frames, additional_info = outputs
        else:
            next_frames = outputs
            additional_info = None

        # Calculate test loss (optional)
        loss = self.MSE_criterion(next_frames[:, :, :, :, 0], frames_tensor[:, 1:, :, :, 0])

        # Optionally process additional_info
        if additional_info is not None:
            pass

        return next_frames.detach().cpu().numpy(), loss.detach().cpu().numpy()
