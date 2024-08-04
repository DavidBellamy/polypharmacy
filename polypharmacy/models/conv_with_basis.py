from typing import Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, Size


class GeneralConvWithBasis(MessagePassing):
    """
    GeneralConv with custom linear layer, which takes in weights and biases as input
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: Optional[int],
        basis_lin_msg_wt,
        basis_lin_msg_biases,
        linear_combinations,
        basis_lin_self_wt=None,
        basis_lin_self_biases=None,
        aggr: str = "add",
        skip_linear: str = False,
        l2_normalize: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("aggr", aggr)
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l2_normalize = l2_normalize
        self.basis_lin_self_wt = basis_lin_self_wt
        self.basis_lin_self_biases = basis_lin_self_biases
        self.basis_lin_msg_wt = basis_lin_msg_wt
        self.basis_lin_msg_biases = basis_lin_msg_biases
        self.linear_combinations = linear_combinations
        self.skip_linear = skip_linear

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

    @torch.jit.export
    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: Tensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        x_self = x[1]

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        if self.skip_linear or self.in_channels != self.out_channels:
            assert (
                self.basis_lin_self_wt is not None
                and self.basis_lin_self_biases is not None
            )
            lin_self_wt = torch.matmul(
                self.basis_lin_self_wt, self.linear_combinations.t()
            ).squeeze()
            lin_self_biases = torch.matmul(
                self.basis_lin_self_biases, self.linear_combinations.t()
            ).squeeze()
            x_self = F.linear(x_self, lin_self_wt.t(), lin_self_biases)
        out = out + x_self
        if self.l2_normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def message_basic(self, x_j: Tensor):
        # TODO: is x_i needed? GeneralConv.message_basic() has it.
        lin_msg_biases = torch.matmul(
            self.basis_lin_msg_biases, self.linear_combinations.t()
        ).squeeze()
        lin_msg_wt = torch.matmul(
            self.basis_lin_msg_wt, self.linear_combinations.t()
        ).squeeze()
        return F.linear(x_j, lin_msg_wt.t(), lin_msg_biases)

    def message(
        self,
        x_j: Tensor,
    ) -> Tensor:
        x_j_out = self.message_basic(x_j)

        return x_j_out

def get_scripted_general_conv_with_basis(*args, **kwargs):
    return torch.jit.script(GeneralConvWithBasis(*args, **kwargs))
