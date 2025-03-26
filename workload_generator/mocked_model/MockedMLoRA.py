"""
MockedMLoRA.py - Mocked implementation of mLoRA for SimAI simulation
"""

from utils.utils import divide, CommType, CommGroup
from workload_generator.mocked_model.MockedModel import MockedModel, Linear, MockedParam
from log_analyzer.log import Workload, LogItem
import torch

# ? QUESTIONS:
# - Do I have to mock the Adapter class (mlora/model/modules/adapter.py) - no right?

class MockedAdapter(MockedModel):
    """Base class for mocking adapters in mLoRA.
    
    Follows the mlora/model/modules/adapter.py > Adapter class
    """
    def __init__(self, adapter_name: str, adapter_type: str):
        super().__init__()
        self.adapter_name_ = adapter_name
        self.adapter_type_ = adapter_type

class MockedLoRA(MockedAdapter):
    """
    Mocks a LoRA adapter in mLoRA.
    Corresponds to the mlora/model/modules/lora.py > LoRA(Adapter) class
    """
    
    def __init__(
        self,
        adapter_name: str,
        in_dim: int,
        out_dim: int,
        r: int,
        alpha: int,
        dropout: float,
    ):
        super().__init__(adapter_name, "lora")
        self.adapter_name: str = adapter_name
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.rank: int = r
        self.alpha: int = alpha
        self.dropout: float = dropout
        
        # LoRA weights - a & b matrices
        self.lora_a: torch.Tensor = MockedParam((r, in_dim), name=f"{adapter_name}_lora_a")
        self.lora_b: torch.Tensor = MockedParam((out_dim, r), name=f"{adapter_name}_lora_b")
        self.scaling: float = alpha / r
        
    # TODO: should I move the forward function logic from the MockedLinearWithLoRA class to here, and then call that function in there?

class MockedDoRA(MockedLoRA):
    """Mocks a DoRA adapter in mLoRA.

    DoRA extends LoRA with weight decomposition. 
    Follows the mlora/model/modules/dora.py > DoRA(LoRA) class
    """
    def __init__(
        self,
        adapter_name: str,
        in_dim: int,
        out_dim: int,
        r: int,
        alpha: int,
        dropout: float,
    ):
        super().__init__(adapter_name, in_dim, out_dim, r, alpha, dropout)
        self.adapter_type_ = "dora" 
        self.magnitude_: MockedParam = MockedParam((1, out_dim), name=f"{adapter_name}_magnitude")

# TODO: mock the VeRA adapter as well from mlora/model/modules/vera.py
    
class MockedBatchLoRA(MockedModel):
    pass # TODO: implement the batch lora operator as discussed in the paper


class MockedLinearWithLoRA(MockedModel):
    """Mocks a linear layer with LoRA adapters. 
    
    Trying to follow the mlora/model/modules/linear.py > Linear class
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_id: int,
        prefix_name: str,
        batch_size: int,
        seq_len: int,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.name = prefix_name
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # Frozen base weights
        self.weight_: MockedParam = MockedParam(
            (out_features, in_features), name=f"{prefix_name}_weight"
        )
        
        self.adapters_: dict = {}  # Stores adapter instances. Key=string being the adapter name, value=Adapter

    def load_adapter(self, adapter):
        """Add an adapter to this linear layer"""
        self.adapters_[adapter.adapter_name_] = adapter
        
    def offload_adapter(self, adapter_name: str):
        """Delete an adapter from this linear layer"""
        if adapter_name not in self.adapters_:
            return

        del self.adapters_[adapter_name]
    
    def forward(self, data: torch.Tensor, adapter_id=None):
        """Forward pass through the frozen linear layer and the lora adapter
        
        adapter_id = Name of the adapter (key to self.adapters_)
        """
        workloads = Workload()
        
        # Base model computation (frozen weights)
        base_computation = self.batch_size * self.seq_len * self.in_features * self.out_features
        workloads.append(
            LogItem(
                comm_type=CommType.computation,
                msg_size=base_computation,
                stage=f"forward.MockedLinearWithLoRA.{self.name}.base",
            )
        )
        
        # Lora adapter computation
        if adapter_id is not None:
            if adapter_id not in self.adapters_:
                return
            workloads.extend(self._process_adapter_forward(adapter_id))
        else:
            # process all adapters then
            for adapter_name, _ in self.adapters_.items():
                workloads.extend(self._process_adapter_forward(adapter_name))

        return workloads

    def backward(self, adapter_id=None):
        """Backward pass for the linear layer with adapters"""
        workloads = Workload()
        
        # No gradient computation for base model (frozen weights)
        
        # Process adapters
        if adapter_id is not None:
            # Process only the specified adapter
            if adapter_id in self.adapters_:
                workloads.extend(self._process_adapter_backward(adapter_id))
        else:
            # Process all adapters
            for adapter_name, _ in self.adapters_.items():
                workloads.extend(self._process_adapter_backward(adapter_name))
        
        return workloads
    
    def _process_adapter_forward(self, adapter_name: str):
        workload = Workload()
        
        # ? should I use adapter_name or adapter_type here? MockedAdapter
        if adapter_name == "lora":
            adapter = self.adapters_[adapter_name]
            # First matmul (result x A)
            # [batch_size, seq_len, in_features] x [in_features, r] -> [batch_size, seq_len, r]
            workload.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=self.batch_size * self.seq_len * self.in_features * adapter.r_,
                    stage=f"forward.MockedLinearWithLoRA.{self.name}.{adapter.adapter_name_}.lora_a",
                )
            )
            
            # Second matmul (result × B)
            # [batch_size, seq_len, r] x [r, out_features] -> [batch_size, seq_len, out_features]
            workload.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=self.batch_size * self.seq_len * adapter.r_ * self.out_features,
                    stage=f"forward.MockedLinearWithLoRA.{self.name}.{adapter.adapter_name_}.lora_b",
                )
            )
        elif adapter_name == "dora":
            raise NotImplementedError()
        elif adapter_name == "vera":
            raise NotImplementedError()
        
        return workload

    def _process_adapter_backward(self, adapter_name: str):

        workloads = Workload()
        
        if adapter_name == "lora":
            adapter = self.adapters_[adapter_name]
            
            # Gradient computation for B (grad_output and drop_data)
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=self.batch_size * self.seq_len * self.out_features * adapter.r_,
                    stage=f"backward.MockedLinearWithLoRA.{self.name}.{adapter.adapter_name_}.grad_b",
                )
            )
            
            # Gradient computation for A (bstage and lora_data)
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=self.batch_size * self.seq_len * adapter.r_ * self.in_features,
                    stage=f"backward.MockedLinearWithLoRA.{self.name}.{adapter.adapter_name_}.grad_a",
                )
            )
            
            # Gradient computation for input data
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=self.batch_size * self.seq_len * self.in_features,
                    stage=f"backward.MockedLinearWithLoRA.{self.name}.{adapter.adapter_name_}.grad_input",
                )
            )
        

class MockedLoRAPP(MockedModel):
    pass # TODO implement

