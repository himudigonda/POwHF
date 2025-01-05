import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from powhf import utils

tkwargs = {
    "device": torch.device("cuda:0"),
    "dtype": torch.float32,
}

class MLPRegression(nn.Module):
    """A simple MLP for regression."""
    def __init__(self, input_dim=86):
        utils.debug_log(f"powhf.LlamaForMLPRegression.MLPRegression :: Initializing MLP, input dim: {input_dim}")
        super(MLPRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        utils.debug_log(f"powhf.LlamaForMLPRegression.MLPRegression.forward :: Forward pass")
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output

class CustomImageDataset(Dataset):
    """A dataset for image data."""
    def __init__(self, X_train, Y_train):
        utils.debug_log(f"powhf.LlamaForMLPRegression.CustomImageDataset :: Initializing dataset, X_train size: {len(X_train)}, Y_train size: {len(Y_train)}")
        self.X_train = X_train
        self.Y_train = Y_train

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]


class ENN(nn.Module):
    def __init__(self, input_dim, hidden_size=32, depth=2, init_params=None):
        super(ENN, self).__init__()
        utils.debug_log(f"powhf.LlamaForMLPRegression.ENN :: Initializing ENN, input dim: {input_dim}, hidden size: {hidden_size}, depth: {depth}")
        self.activate = nn.ReLU()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, hidden_size))
        for i in range(depth-1):
            self.layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.layer_list.append(nn.Linear(hidden_size, 1))

        self.layer_list_10 = nn.ModuleList()
        for i in range(10):
            new_module = nn.ModuleList()
            new_module.append(nn.Linear(input_dim, hidden_size))
            for j in range(depth-1):
                new_module.append(nn.Linear(hidden_size, hidden_size))
            new_module.append(nn.Linear(hidden_size, 1))
            self.layer_list_10.append(new_module)

        if init_params is None:
            for i in range(len(self.layer_list)):
                torch.nn.init.normal_(self.layer_list[i].weight, mean=0, std=1.0)
                torch.nn.init.normal_(self.layer_list[i].bias, mean=0, std=1.0)
        else:
            for i in range(len(self.layer_list)):
                self.layer_list[i].weight.data = init_params[i*2]
                self.layer_list[i].bias.data = init_params[i*2+1]

        for i in range(10):
            for j in range(len(self.layer_list)):
                self.layer_list_10[i][j].weight.data.copy_(self.layer_list[j].weight.data)
                self.layer_list_10[i][j].bias.data.copy_(self.layer_list[j].bias.data)

        for param in self.layer_list.parameters():
            param.requires_grad = False

    def forward(self, x, idx):
        y = x
        for i in range(len(self.layer_list_10[idx])-1):
            y = self.activate(self.layer_list_10[idx][i](y))
        y = self.layer_list_10[idx][-1](y)
        return y

class DoubleTS:
    def __init__(self, input_dim, lamdba=1, nu=1, style='ucb', init_x=None, init_y=None, diagonalize=True):
        utils.debug_log(f"powhf.LlamaForMLPRegression.DoubleTS :: Initializing DoubleTS, input dim: {input_dim}, lambda: {lamdba}, nu: {nu}, style: {style}")
        self.diagonalize = diagonalize
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.func = ENN(input_dim).to(**tkwargs)
        self.init_state_dict = deepcopy(self.func.state_dict())

        if init_x is not None:
            self.pair_embedding = init_x.to(**tkwargs)
        else:
            self.pair_embedding = None
        if init_y is not None:
            self.reward = init_y.to(**tkwargs).to(dtype=torch.int64)
        else:
            self.reward = None
        self.len = 0
        self.lamdba = lamdba

        self.nu = nu
        self.lamdba = lamdba
        self.style = style
        self.mean = None
        self.std = None


    def select(self, context, select_idx_history, prompt_domain_id=None, batch_size=300):
        utils.debug_log(f"powhf.LlamaForMLPRegression.DoubleTS.select :: Selecting arms, context size: {context.shape[0]}")
        context_size = context.shape[0]
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)

        final_arms = []
        up_k = 5
        k_ = 0
        while len(final_arms) < 2:
            mu = []
            self.func.train()
            epi_idx = torch.randint(0, 10, (1,))
            for i in range(n_batchs):
                if i == n_batchs - 1:
                    context_batch = context[(i*batch_size):]
                else:
                    context_batch = context[(i*batch_size):((i+1)*batch_size)]

                mu_ = self.func(context_batch, epi_idx)

                mu.append(mu_.cpu())
            mu = torch.vstack(mu)

            if prompt_domain_id is None:
                arm1 = torch.argmax(mu.view(-1))
            else:
                arm1_ = torch.argmax(mu.view(-1)[prompt_domain_id])
                prompt_domain_id_ = torch.tensor(prompt_domain_id)
                arm1 = prompt_domain_id_[arm1_]

            if arm1 not in final_arms:
                final_arms.append(arm1)
            else:
                k_ += 1
            if k_ > up_k:
                if prompt_domain_id is None:
                    random_arm = torch.randint(0, context_size, (2,))
                else:
                    prompt_domain_id_ = torch.tensor(prompt_domain_id)
                    random_arm = torch.randint(0, len(prompt_domain_id), (2,))
                    random_arm = prompt_domain_id_[random_arm]
                if random_arm[0] not in final_arms:
                    final_arms.append(random_arm[0])
                else:
                    final_arms.append(random_arm[1])
                break
        utils.debug_log(f"powhf.LlamaForMLPRegression.DoubleTS.select :: Selected arms: {final_arms[0]}, {final_arms[1]}")
        return final_arms[0], final_arms[1]


    def find_best(self, context, select_idx_history, all_domain=False, batch_size=300):
        utils.debug_log(f"powhf.LlamaForMLPRegression.DoubleTS.find_best :: Finding best arm, context size: {context.shape[0]}, all domain: {all_domain}")
        context_size = context.shape[0]
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        mu = []
        self.func.eval()
        for i in range(n_batchs):
            epi_idx = torch.randint(0, 10, (1,))
            if i == n_batchs - 1:
                context_batch = context[(i*batch_size):]
            else:
                context_batch = context[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func(context_batch, epi_idx)
            mu.append(mu_.cpu())
        mu = torch.vstack(mu)

        if all_domain:
            arm = torch.argmax(mu.view(-1))
        else:
            all_queried = torch.tensor(select_idx_history).view(-1)
            arm_ = torch.argmax(mu.view(-1)[all_queried])
            arm = all_queried[arm_]
        utils.debug_log(f"powhf.LlamaForMLPRegression.DoubleTS.find_best :: Best arm: {arm}")
        return arm


    def train(self, context=None, reward=None, local_training_iter=30):
        utils.debug_log(f"powhf.LlamaForMLPRegression.DoubleTS.train :: Training model, local iterations: {local_training_iter}")
        if self.init_state_dict is not None:
            self.func.load_state_dict(deepcopy(self.init_state_dict))
        if context is not None:
            self.pair_embedding = torch.cat((self.pair_embedding, context.reshape(2, 1, -1).to(**tkwargs)), dim=1)
            self.reward = torch.cat((self.reward, torch.tensor([reward]).reshape(-1).to(**tkwargs).to(dtype=torch.int64)))

        self.len = self.pair_embedding.shape[1]
        optimizer = torch.optim.Adam(self.func.parameters(), lr=1e-3)
        batch_size = 32
        if self.len < batch_size:
            lamdba_ = self.lamdba
        else:
            lamdba_ = self.lamdba * batch_size / (self.len)
        self.func.train()
        reward_ = 1 - self.reward.reshape(-1)
        for _ in range(local_training_iter):
            if self.len // batch_size == 0:
                selected_idx = torch.arange(0, self.len)
                epi_idx = torch.randint(0, 10, (1,))
                self.func.zero_grad()
                optimizer.zero_grad()
                side_1 = self.pair_embedding[0, selected_idx, :].reshape(len(selected_idx), -1)
                side_2 = self.pair_embedding[1, selected_idx, :].reshape(len(selected_idx), -1)
                pred_1 = self.func(side_1, epi_idx)
                pred_2 = self.func(side_2, epi_idx)
                ce_ = torch.mean(-(1-reward_[selected_idx].to(dtype=torch.float32)) * pred_1.reshape(-1) - reward_[selected_idx].to(dtype=torch.float32) * pred_2.reshape(-1) + torch.log(torch.exp(pred_1.reshape(-1)) + torch.exp(pred_2.reshape(-1))))
                l2_reg_term = 0
                for param1, param2 in zip(self.func.layer_list_10[epi_idx], self.func.layer_list):
                    l2_reg_term += torch.sum((param1.weight - param2.weight) ** 2) + torch.sum((param1.bias - param2.bias) ** 2)
                loss = ce_ + lamdba_ * l2_reg_term
                loss.backward()
                optimizer.step()
            else:
                for batch_id in range((self.len // batch_size)):
                    if batch_id == (self.len // batch_size) - 1 and self.len % batch_size != 0:
                        selected_idx = torch.arange(batch_id*batch_size, self.len)
                    else:
                        selected_idx = torch.arange(batch_id*batch_size, (batch_id+1)*batch_size)
                    epi_idx = torch.randint(0, 10, (1,))
                    self.func.zero_grad()
                    optimizer.zero_grad()
                    side_1 = self.pair_embedding[0, selected_idx, :].reshape(len(selected_idx), -1)
                    side_2 = self.pair_embedding[1, selected_idx, :].reshape(len(selected_idx), -1)
                    pred_1 = self.func(side_1, epi_idx)
                    pred_2 = self.func(side_2, epi_idx)
                    ce_ = torch.mean(-(1-reward_[selected_idx].to(dtype=torch.float32)) * pred_1.reshape(-1) - reward_[selected_idx].to(dtype=torch.float32) * pred_2.reshape(-1) + torch.log(torch.exp(pred_1.reshape(-1)) + torch.exp(pred_2.reshape(-1))))
                    l2_reg_term = 0
                    for param1, param2 in zip(self.func.layer_list_10[epi_idx], self.func.layer_list):
                        l2_reg_term += torch.sum((param1.weight - param2.weight) ** 2) + torch.sum((param1.bias - param2.bias) ** 2)
                    loss = ce_ + lamdba_ * l2_reg_term
                    loss.backward()
                    optimizer.step()
        utils.debug_log(f"powhf.LlamaForMLPRegression.DoubleTS.train :: Training Loss : {loss.item()}")
        return self.func.state_dict()

class Network(nn.Module):
    def __init__(self, input_dim, hidden_size=32, depth=2, init_params=None):
        super(Network, self).__init__()
        utils.debug_log(f"powhf.LlamaForMLPRegression.Network :: Initializing Network, input dim: {input_dim}, hidden size: {hidden_size}, depth: {depth}")
        self.activate = nn.ReLU()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, hidden_size))
        for i in range(depth-1):
            self.layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.layer_list.append(nn.Linear(hidden_size, 1))

        if init_params is None:
            for i in range(len(self.layer_list)):
                torch.nn.init.normal_(self.layer_list[i].weight, mean=0, std=1.0)
                torch.nn.init.normal_(self.layer_list[i].bias, mean=0, std=1.0)
        else:
            for i in range(len(self.layer_list)):
                self.layer_list[i].weight.data = init_params[i*2]
                self.layer_list[i].bias.data = init_params[i*2+1]

    def forward(self, x):
        y = x
        for i in range(len(self.layer_list)-1):
            y = self.activate(self.layer_list[i](y))
        y = self.layer_list[-1](y)
        return y

class NeuralDBDiag:
    def __init__(self, input_dim, lamdba=1, nu=1, style='ucb', init_x=None, init_y=None, diagonalize=True):
        utils.debug_log(f"powhf.LlamaForMLPRegression.NeuralDBDiag :: Initializing NeuralDBDiag, input dim: {input_dim}, lambda: {lamdba}, nu: {nu}, style: {style}")
        self.diagonalize = diagonalize
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.func = extend(Network(input_dim).to(**tkwargs))
        self.init_state_dict = deepcopy(self.func.state_dict())

        if init_x is not None:
            self.pair_embedding = init_x.to(**tkwargs)
        else:
            self.pair_embedding = None
        if init_y is not None:
            self.reward = init_y.to(**tkwargs).to(dtype=torch.int64)
        else:
            self.reward = None
        self.len = 0
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)

        self.nu = nu
        self.lamdba = lamdba
        self.style = style
        self.loss_func = nn.MSELoss()
        self.mean = None
        self.std = None


    def select(self, context, select_idx_history, prompt_domain_id=None, batch_size=300):
        utils.debug_log(f"powhf.LlamaForMLPRegression.NeuralDBDiag.select :: Selecting arms, context size: {context.shape[0]}")
        context_size = context.shape[0]
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        g_list = []
        mu = []
        self.func.train()
        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context[(i*batch_size):]
            else:
                context_batch = context[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func(context_batch)
            sum_mu = torch.sum(mu_)
            with backpack(BatchGrad()):
                sum_mu.backward()
            g_list_ = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1)
            g_list.append(g_list_.cpu())
            mu.append(mu_.cpu())
        g_list = torch.vstack(g_list)
        mu = torch.vstack(mu)

        if prompt_domain_id is None:
            if self.nu == -1:
                arm1 = torch.randint(0, context_size, (1,))
            else:
                arm1 = torch.argmax(mu.view(-1))
        else:
            if self.nu == -1:
                arm1 = torch.randint(0, len(prompt_domain_id), (1,))
                prompt_domain_id_ = torch.tensor(prompt_domain_id)
                arm1 = prompt_domain_id_[arm1]
            else:
                arm1_ = torch.argmax(mu.view(-1)[prompt_domain_id])
                prompt_domain_id_ = torch.tensor(prompt_domain_id)
                arm1 = prompt_domain_id_[arm1_]

        history = torch.tensor(select_idx_history)
        grad_1 = g_list[history[:,0]]
        grad_2 = g_list[history[:,1]]
        feature = grad_1 - grad_2

        U = torch.matmul(feature.transpose(0,1), feature) + self.lamdba * torch.eye(self.total_param)
        U = U.diagonal()
        grad_arm_1 = g_list[arm1]
        feature_arm_2 = g_list - grad_arm_1

        sigma = torch.sqrt(torch.sum(self.nu * feature_arm_2 * feature_arm_2 / U, dim=1))
        sample_r = mu.view(-1) + sigma.view(-1)

        if prompt_domain_id is None:
            if self.nu == -1:
                 sorted_idx = torch.argsort(torch.rand(context_size), descending=True)
            else:
                sorted_idx = torch.argsort(sample_r, descending=True)
        else:
            if self.nu == -1:
                sorted_idx = torch.argsort(torch.rand(len(prompt_domain_id)), descending=True)
                prompt_domain_id_ = torch.tensor(prompt_domain_id)
                sorted_idx = prompt_domain_id_[sorted_idx]
            else:
                sorted_idx_ = torch.argsort(sample_r[prompt_domain_id], descending=True)
                prompt_domain_id_ = torch.tensor(prompt_domain_id)
                sorted_idx = prompt_domain_id_[sorted_idx_]
        if sorted_idx[0] == arm1:
            arm2 = sorted_idx[1]
        else:
            arm2 = sorted_idx[0]
        utils.debug_log(f"powhf.LlamaForMLPRegression.NeuralDBDiag.select :: Selected arms: {arm1}, {arm2}")
        return arm1, arm2


    def find_best(self, context, select_idx_history, all_domain=False, batch_size=300):
        utils.debug_log(f"powhf.LlamaForMLPRegression.NeuralDBDiag.find_best :: Finding best arm, all_domain: {all_domain}")
        context_size = context.shape[0]
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        mu = []
        self.func.eval()
        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context[(i*batch_size):]
            else:
                context_batch = context[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func(context_batch)
            mu.append(mu_.cpu())
        mu = torch.vstack(mu)

        if all_domain:
            arm = torch.argmax(mu.view(-1))
        else:
            all_queried = torch.tensor(select_idx_history).view(-1)
            arm_ = torch.argmax(mu.view(-1)[all_queried])
            arm = all_queried[arm_]
        utils.debug_log(f"powhf.LlamaForMLPRegression.NeuralDBDiag.find_best :: Best arm: {arm}")
        return arm


    def train(self, context=None, reward=None, local_training_iter=30):
        utils.debug_log(f"powhf.LlamaForMLPRegression.NeuralDBDiag.train :: Training model, local iterations: {local_training_iter}")
        if self.init_state_dict is not None:
            self.func.load_state_dict(deepcopy(self.init_state_dict))
        if context is not None:
            self.pair_embedding = torch.cat((self.pair_embedding, context.reshape(2, 1, -1).to(**tkwargs)), dim=1)
            self.reward = torch.cat((self.reward, torch.tensor([reward]).reshape(-1).to(**tkwargs).to(dtype=torch.int64)))

        self.len = self.pair_embedding.shape[1]
        optimizer = torch.optim.Adam(self.func.parameters(), lr=1e-3, weight_decay=self.lamdba / (self.len+50))
        self.func.train()
        for _ in range(local_training_iter):
            self.func.zero_grad()
            optimizer.zero_grad()
            side_1 = self.pair_embedding[0].reshape(self.len, -1)
            side_2 = self.pair_embedding[1].reshape(self.len, -1)
            pred_1 = self.func(side_1)
            pred_2 = self.func(side_2)
            logits = (pred_1 - pred_2).reshape(-1)
            reward_ = self.reward.reshape(-1)
            loss = F.binary_cross_entropy_with_logits(logits, reward_.to(dtype=torch.float32))
            loss.backward()
            optimizer.step()
        utils.debug_log(f"powhf.LlamaForMLPRegression.NeuralDBDiag.train :: Training loss: {loss.item()}")
        return self.func.state_dict()

class LinearModel(nn.Module):
    def __init__(self, input_dim, init_params=None):
        utils.debug_log(f"powhf.LlamaForMLPRegression.LinearModel :: Initializing LinearModel, input dim: {input_dim}")
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        y = self.linear(x)
        return y


class LinearDBDiag:
    def __init__(self, input_dim, lamdba=1, nu=1, style='ucb', init_x=None, init_y=None, diagonalize=True):
        utils.debug_log(f"powhf.LlamaForMLPRegression.LinearDBDiag :: Initializing LinearDBDiag, input dim: {input_dim}, lambda: {lamdba}, nu: {nu}, style: {style}")
        self.diagonalize = diagonalize
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.func = LinearModel(input_dim).to(**tkwargs)
        self.init_state_dict = deepcopy(self.func.state_dict())

        if init_x is not None:
            self.pair_embedding = init_x.to(**tkwargs)
        else:
            self.pair_embedding = None
        if init_y is not None:
            self.reward = init_y.to(**tkwargs).to(dtype=torch.int64)
        else:
            self.reward = None
        self.len = 0
        self.lamdba = lamdba
        self.total_param = input_dim

        self.nu = nu
        self.lamdba = lamdba
        self.style = style
        self.loss_func = nn.MSELoss()
        self.mean = None
        self.std = None

    def select(self, context, select_idx_history, prompt_domain_id=None, batch_size=300):
        utils.debug_log(f"powhf.LlamaForMLPRegression.LinearDBDiag.select :: Selecting arms, context size: {context.shape[0]}")
        context_size = context.shape[0]
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        mu = []
        self.func.eval()
        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context[(i*batch_size):]
            else:
                context_batch = context[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func(context_batch)
            mu.append(mu_.cpu())
        mu = torch.vstack(mu)

        if prompt_domain_id is None:
            arm1 = torch.argmax(mu.view(-1))
        else:
            arm1_ = torch.argmax(mu.view(-1)[prompt_domain_id])
            prompt_domain_id_ = torch.tensor(prompt_domain_id)
            arm1 = prompt_domain_id_[arm1_]

        history = torch.tensor(select_idx_history)
        grad_1 = context[history[:,0]]
        grad_2 = context[history[:,1]]
        feature = (grad_1 - grad_2).cpu()

        U = torch.matmul(feature.transpose(0,1), feature)
        U = U.diagonal()
        U = U + 1e-10

        grad_arm_1 = context[arm1]
        feature_arm_2 = (context - grad_arm_1).cpu()

        sigma = torch.sqrt(torch.sum(self.nu * feature_arm_2 * feature_arm_2 / U, dim=1))
        sample_r = mu.view(-1) + sigma.view(-1)

        if prompt_domain_id is None:
            sorted_idx = torch.argsort(sample_r, descending=True)
        else:
            sorted_idx_ = torch.argsort(sample_r[prompt_domain_id], descending=True)
            prompt_domain_id_ = torch.tensor(prompt_domain_id)
            sorted_idx = prompt_domain_id_[sorted_idx_]
        if sorted_idx[0] == arm1:
            arm2 = sorted_idx[1]
        else:
            arm2 = sorted_idx[0]
        utils.debug_log(f"powhf.LlamaForMLPRegression.LinearDBDiag.select :: Selected arms: {arm1}, {arm2}")
        return arm1, arm2

    def find_best(self, context, select_idx_history, all_domain=False, batch_size=300):
        utils.debug_log(f"powhf.LlamaForMLPRegression.LinearDBDiag.find_best :: Finding best arm, all_domain: {all_domain}")
        context_size = context.shape[0]
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        mu = []
        self.func.eval()
        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context[(i*batch_size):]
            else:
                context_batch = context[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func(context_batch)
            mu.append(mu_.cpu())
        mu = torch.vstack(mu)

        if all_domain:
            arm = torch.argmax(mu.view(-1))
        else:
            all_queried = torch.tensor(select_idx_history).view(-1)
            arm_ = torch.argmax(mu.view(-1)[all_queried])
            arm = all_queried[arm_]
        utils.debug_log(f"powhf.LlamaForMLPRegression.LinearDBDiag.find_best :: Best arm: {arm}")
        return arm

    def train(self, context=None, reward=None, local_training_iter=30):
        utils.debug_log(f"powhf.LlamaForMLPRegression.LinearDBDiag.train :: Training model, local iterations: {local_training_iter}")
        if self.init_state_dict is not None:
            self.func.load_state_dict(deepcopy(self.init_state_dict))
        if context is not None:
            self.pair_embedding = torch.cat((self.pair_embedding, context.reshape(2, 1, -1).to(**tkwargs)), dim=1)
            self.reward = torch.cat((self.reward, torch.tensor([reward]).reshape(-1).to(**tkwargs).to(dtype=torch.int64)))

        self.len = self.pair_embedding.shape[1]
        optimizer = torch.optim.Adam(self.func.parameters(), lr=1e-3)
        self.func.train()
        for _ in range(local_training_iter):
            self.func.zero_grad()
            optimizer.zero_grad()
            side_1 = self.pair_embedding[0].reshape(self.len, -1)
            side_2 = self.pair_embedding[1].reshape(self.len, -1)
            pred_1 = self.func(side_1)
            pred_2 = self.func(side_2)
            logits = (pred_1 - pred_2).reshape(-1)
            reward_ = self.reward.reshape(-1)
            loss = F.binary_cross_entropy_with_logits(logits, reward_.to(dtype=torch.float32))

            loss.backward()
            optimizer.step()
        utils.debug_log(f"powhf.LlamaForMLPRegression.LinearDBDiag.train :: Training loss: {loss.item()}")
        return self.func.state_dict()


class MLPRegression_Train:
    def __init__(
        self,
        input_dim=4096,
        optimizer_fn=torch.optim.Adam,
        loss_fn=nn.MSELoss,
        lr=0.001,
        batch_size=64,
        epochs=30,
        device=None,
    ):
        utils.debug_log(
            f"powhf.LlamaForMLPRegression.MLPRegression_Train :: Initializing MLPRegression_Train, input_dim: {input_dim}, lr: {lr}, batch_size: {batch_size}, epochs: {epochs}"
        )

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.model = MLPRegression(input_dim).to(device)
        self.optimizer = optimizer_fn(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn()
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        # backup for the initial model weight and optimizer
        self.init_model_weight = deepcopy(self.model.state_dict())
        self.optimizer_fn = optimizer_fn

    def get_data_loader(self, X_train, Y_train):
        """Creates a DataLoader for the training data."""
        utils.debug_log(
            f"powhf.LlamaForMLPRegression.MLPRegression_Train.get_data_loader :: Creating DataLoader, X_train size: {len(X_train)}, Y_train size: {len(Y_train)}"
        )
        dataset = CustomImageDataset(X_train, Y_train)
        train_dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )
        return train_dataloader

    def fit(self, X_train, Y_train, verbose=False, epochs=None):
        """Fits the MLP regression model."""
        if epochs == None:
            epochs = self.epochs
        utils.debug_log(
            f"powhf.LlamaForMLPRegression.MLPRegression_Train.fit :: Starting training, epochs: {epochs}"
        )

        train_loader = self.get_data_loader(X_train, Y_train)
        for e in range(epochs):
            self.model.train()

            for batch_idx, batch in enumerate(train_loader):
                data, label = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(data)
                loss = self.loss_fn(pred, label)
                loss.backward()
                self.optimizer.step()

            if verbose:
                utils.debug_log(
                    f"powhf.LlamaForMLPRegression.MLPRegression_Train.fit :: Epoch: {e}, Loss: {loss.item():.4f}"
                )

        utils.debug_log(
            "powhf.LlamaForMLPRegression.MLPRegression_Train.fit :: Finished training"
        )
        return self.model

    def select(
        self,
        context,
        diagonalize,
        lamdba,
        nu,
        style,
    ):
        """Selects an arm using the trained model and uncertainty estimates."""
        utils.debug_log(
            f"powhf.LlamaForMLPRegression.MLPRegression_Train.select :: Selecting arm, diagonalize: {diagonalize}, lambda: {lamdba}, nu: {nu}, style: {style}"
        )
        self.model.train()
        mu = self.model(context)
        sum_mu = torch.sum(mu)
        with backpack(BatchGrad()):
            sum_mu.backward()

        g_list = torch.cat(
            [
                p.grad_batch.flatten(start_dim=1).detach()
                for p in self.func.parameters()
            ],
            dim=1,
        )

        if diagonalize:
            sigma = torch.sqrt(torch.sum(lamdba * nu * g_list * g_list / self.U, dim=1))
        else:
            tmp = torch.matmul(g_list, torch.inverse(self.U))
            sigma = torch.sqrt(
                nu * lamdba * torch.matmul(tmp, torch.transpose(g_list, 0, 1))
            )
            sigma = torch.diagonal(sigma, 0)

        if style == "ts":
            sample_r = torch.normal(mu.view(-1), sigma.view(-1))
        elif style == "ucb":
            sample_r = mu.view(-1) + sigma.view(-1)
        arm = torch.argmax(sample_r)
        if diagonalize:
            self.U += g_list[arm] * g_list[arm]
        else:
            self.U += torch.outer(g_list[arm], g_list[arm])

        utils.debug_log(
            f"powhf.LlamaForMLPRegression.MLPRegression_Train.select :: Selected arm: {arm.item()}, grad norm: {g_list[arm].norm().item()}"
        )
        return arm, g_list[arm].norm().item()

    def restart_model(self):
        """Restores the model to its initial state."""
        utils.debug_log(
            "powhf.LlamaForMLPRegression.MLPRegression_Train.restart_model :: Restarting model"
        )
        self.model.load_state_dict(deepcopy(self.init_model_weight))
        self.optimizer = self.optimizer_fn(self.model.parameters(), lr=self.lr)
