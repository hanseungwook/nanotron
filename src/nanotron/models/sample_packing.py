import torch
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple, List

# Import flash_attn functions (requires flash-attn installed)
from flash_attn import flash_attn_func, flash_attn_varlen_func

# Set default dtype to bfloat16 as FlashAttention only supports fp16 and bf16
torch.set_default_dtype(torch.bfloat16)


class DocumentAwareFlashAttention(torch.nn.Module):
    """
    An implementation of document-aware Flash Attention that uses position IDs
    to create document masks for packed sequences.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_prob: float = 0.0, causal: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_prob = dropout_prob
        self.causal = causal

        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_probs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for document-aware Flash Attention.

        Args:
            q: Query tensor of shape [batch_size, seq_len, hidden_size] or [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor of shape [batch_size, seq_len, hidden_size] or [batch_size, seq_len, num_heads, head_dim]
            v: Value tensor of shape [batch_size, seq_len, hidden_size] or [batch_size, seq_len, num_heads, head_dim]
            position_ids: Optional tensor of shape [batch_size, seq_len] for marking positions in each document
            document_ids: Optional tensor of shape [batch_size, seq_len] for marking document boundaries
            attention_mask: Optional attention mask (will be used if position_ids and document_ids are None)
            return_attn_probs: Whether to return attention probabilities

        Returns:
            output: Attention output of shape [batch_size, seq_len, hidden_size]
            attn_probs: Optional attention probabilities if return_attn_probs is True
        """
        # Check if we have document boundaries defined
        using_document_masking = position_ids is not None or document_ids is not None

        batch_size, seq_len = q.shape[0], q.shape[1]

        # Reshape inputs if needed
        if q.ndim == 3:  # [batch_size, seq_len, hidden_size]
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Rearrange to flash attention expected format: [batch_size, seq_len, num_heads, head_dim]
        if q.shape[2] != self.num_heads:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        if using_document_masking:
            # If we have position_ids but no document_ids, derive document_ids from position IDs
            if document_ids is None and position_ids is not None:
                document_ids = self._derive_document_ids_from_position_ids(position_ids)

            # Create cumulative sequence lengths for each document
            cu_seqlens, max_seqlen = self._get_cu_seqlens(document_ids)

            # Use flash_attn_varlen_func for variable length sequences with document masking
            output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,  # Same for query and key/value
                max_seqlen,
                max_seqlen,  # Same max length for query and key/value
                dropout_p=self.dropout_prob,
                softmax_scale=self.scale,
                causal=self.causal,
            )

            # Reshape output back to [batch_size, seq_len, hidden_size]
            output = output.reshape(batch_size, seq_len, self.hidden_size)

            # We can't efficiently get attention probabilities when using flash_attn_varlen_func
            attn_probs = None

        else:
            # Standard flash attention with or without attention mask
            if attention_mask is not None:
                # Apply attention mask if provided
                # For flash_attn_func, we would convert attention_mask to the appropriate form
                # However, flash_attn_func doesn't directly support attention masks
                # So we use the standard attention mechanism for this case
                q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                attn_output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attention_mask, dropout_p=self.dropout_prob, is_causal=self.causal
                )

                output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
                attn_probs = None  # F.scaled_dot_product_attention does not return attn probs
            else:
                # Use flash_attn_func for standard attention without masking
                output = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout_prob,
                    softmax_scale=self.scale,
                    causal=self.causal,
                )

                # Reshape output back to [batch_size, seq_len, hidden_size]
                output = output.reshape(batch_size, seq_len, self.hidden_size)
                attn_probs = None

        if return_attn_probs:
            return output, attn_probs
        else:
            return output

    def _derive_document_ids_from_position_ids(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Derive document IDs from position IDs.

        When position IDs reset (go back to zero or a small value after being larger),
        it indicates a document boundary.

        Args:
            position_ids: Position IDs of shape [batch_size, seq_len] or [1, total_seq_len]

        Returns:
            document_ids: Document IDs of shape [batch_size, seq_len] or [1, total_seq_len]
        """
        # Handle the case where position_ids is [1, total_seq_len] (for packed sequences)
        if position_ids.shape[0] == 1:
            # Flatten to [total_seq_len]
            flat_position_ids = position_ids.view(-1)

            # Initialize document IDs
            document_ids = torch.zeros_like(flat_position_ids)

            # Identify where positions reset (indicating a new document)
            # We look for positions where the current position is less than the previous
            # This works because positions restart from 0 (or a small value) at document boundaries
            resets = torch.cat(
                [
                    torch.tensor([0], device=position_ids.device, dtype=torch.long),
                    (flat_position_ids[1:] <= flat_position_ids[:-1]).long(),
                ]
            )

            # Cumulative sum to assign document IDs
            document_ids = torch.cumsum(resets, dim=0)

            # Reshape back to original shape
            document_ids = document_ids.view(position_ids.shape)

        else:
            # For standard batched position_ids [batch_size, seq_len]
            # Each item in the batch is a different document, so document_ids is just the batch indices
            batch_size, seq_len = position_ids.shape
            document_ids = torch.arange(batch_size, device=position_ids.device)
            document_ids = document_ids.view(-1, 1).expand(-1, seq_len)

        return document_ids

    def _get_cu_seqlens(self, document_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Get cumulative sequence lengths for each document from document IDs.

        Args:
            document_ids: Document IDs of shape [batch_size, seq_len] or [1, total_seq_len]

        Returns:
            cu_seqlens: Cumulative sequence lengths tensor of shape [num_docs + 1]
            max_seqlen: Maximum sequence length in any document
        """
        # Flatten document IDs
        flat_doc_ids = document_ids.view(-1)

        # Count tokens in each document
        unique_doc_ids, doc_counts = torch.unique_consecutive(flat_doc_ids, return_counts=True)

        # Create cu_seqlens: [0, len1, len1+len2, ...]
        cu_seqlens = torch.zeros(len(unique_doc_ids) + 1, device=document_ids.device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(doc_counts, dim=0)

        # Get max sequence length in any document
        max_seqlen = doc_counts.max().item()

        return cu_seqlens, max_seqlen


# Data Collator for packing sequences with position IDs
class PositionIdAwareCollator:
    """
    Collates examples without padding, using position_ids to mark document boundaries.
    """

    def __init__(self, pad_token_id=-100):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[dict]) -> dict:
        """
        Pack examples together and assign position IDs to mark document boundaries.

        Args:
            features: List of feature dictionaries, each containing 'input_ids' and
                     optionally 'labels' and other fields

        Returns:
            Dict with packed 'input_ids', 'labels', and 'position_ids'
        """
        input_ids = []
        labels = []
        position_ids = []

        for feature in features:
            # Get sequence length
            seq_length = len(feature["input_ids"])

            # Add input IDs
            input_ids.extend(feature["input_ids"])

            # Add labels if they exist
            if "labels" in feature:
                feature_labels = list(feature["labels"])
                # Convert first label to pad_token_id to mark document boundary
                if len(feature_labels) > 0:
                    feature_labels[0] = self.pad_token_id
                labels.extend(feature_labels)

            # Create position IDs starting from 0 for each document
            doc_position_ids = list(range(seq_length))
            position_ids.extend(doc_position_ids)

        # Convert to tensors
        input_ids = torch.tensor([input_ids])
        labels = torch.tensor([labels]) if labels else None
        position_ids = torch.tensor([position_ids])

        return {"input_ids": input_ids, "labels": labels, "position_ids": position_ids}


# Testing Functions
def test_document_masking_with_position_ids():
    """
    Test document-aware Flash Attention with position IDs for packed sequences.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    device = torch.device("cuda")

    # Ensure we're using bfloat16 for Flash Attention compatibility
    print(f"Using dtype: {torch.get_default_dtype()}")

    # Parameters
    batch_size = 1  # We'll use a single batch for packed sequences
    hidden_size = 128
    num_heads = 4
    head_dim = hidden_size // num_heads

    # Create packed sequences - simulating 3 documents of different lengths
    seq_lengths = [3, 5, 7]
    total_seq_len = sum(seq_lengths)

    # Generate fake input IDs for 3 packed sequences
    input_ids = torch.randint(0, 1000, (batch_size, total_seq_len), device=device)

    # Create position IDs that reset at document boundaries
    position_ids = torch.zeros((batch_size, total_seq_len), dtype=torch.long, device=device)

    # Populate position IDs: [0,1,2, 0,1,2,3,4, 0,1,2,3,4,5,6]
    current_pos = 0
    for seq_len in seq_lengths:
        for i in range(seq_len):
            position_ids[0, current_pos + i] = i
        current_pos += seq_len

    # Create QKV for testing - using bfloat16 for Flash Attention compatibility
    q = torch.randn(batch_size, total_seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, total_seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, total_seq_len, hidden_size, device=device, dtype=torch.bfloat16)

    # Create our document-aware attention module
    attn = DocumentAwareFlashAttention(hidden_size, num_heads, dropout_prob=0.0)
    attn = attn.to(device)

    # Initialize expected outputs
    # We'll manually compute attention for each document separately and combine
    expected_output = torch.zeros_like(q)

    # Create a baseline standard attention for comparison
    def standard_attention(q, k, v, mask=None):
        q = q.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        # Compute attention scores
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [batch, heads, seq, seq]

        # Apply attention mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attn_probs = torch.nn.functional.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, hidden_size)

        return out, attn_probs

    # Create document mask for comparison
    document_mask = torch.zeros((batch_size, total_seq_len, total_seq_len), device=device)
    start_idx = 0
    for seq_len in seq_lengths:
        document_mask[0, start_idx : start_idx + seq_len, start_idx : start_idx + seq_len] = 1
        start_idx += seq_len

    # Compute expected output using standard attention with document mask
    standard_out, _ = standard_attention(q, k, v, document_mask)

    # Compute output using our Flash Attention implementation
    flash_out = attn(q, k, v, position_ids=position_ids)

    # Verify that outputs are close
    # Note: There will be numerical differences between standard attention and Flash Attention
    # due to different computation order, but they should be close
    tolerance = 1e-3
    error = (standard_out - flash_out).abs().mean().item()
    print(f"Mean absolute error between standard and flash attention: {error}")
    assert error < tolerance, f"Error too large: {error} > {tolerance}"

    # Test deriving document IDs from position IDs
    document_ids = attn._derive_document_ids_from_position_ids(position_ids)

    # Expected document IDs: [0,0,0, 1,1,1,1,1, 2,2,2,2,2,2,2]
    expected_document_ids = torch.zeros_like(position_ids)
    current_pos = 0
    for doc_idx, seq_len in enumerate(seq_lengths):
        expected_document_ids[0, current_pos : current_pos + seq_len] = doc_idx
        current_pos += seq_len

    document_ids_correct = torch.all(document_ids == expected_document_ids).item()
    print(f"Document IDs derived correctly: {document_ids_correct}")
    assert document_ids_correct, "Document IDs not derived correctly"

    # Test creating cumulative sequence lengths
    cu_seqlens, max_seqlen = attn._get_cu_seqlens(document_ids)

    # Expected cu_seqlens: [0, 3, 8, 15]
    expected_cu_seqlens = torch.tensor([0, 3, 8, 15], dtype=torch.int32, device=device)
    expected_max_seqlen = 7  # Longest sequence is 7

    cu_seqlens_correct = torch.all(cu_seqlens == expected_cu_seqlens).item()
    max_seqlen_correct = max_seqlen == expected_max_seqlen

    print(f"Cumulative sequence lengths correct: {cu_seqlens_correct}")
    print(f"Max sequence length correct: {max_seqlen_correct}")

    assert cu_seqlens_correct, "Cumulative sequence lengths not correct"
    assert max_seqlen_correct, "Max sequence length not correct"

    print("All tests passed!")


def test_position_id_aware_collator():
    """Test the PositionIdAwareCollator for packing sequences."""
    # Create some example features
    features = [
        {"input_ids": [1, 2, 3], "labels": [10, 20, 30]},
        {"input_ids": [4, 5, 6, 7], "labels": [40, 50, 60, 70]},
        {"input_ids": [8, 9], "labels": [80, 90]},
    ]

    # Create collator
    collator = PositionIdAwareCollator(pad_token_id=-100)

    # Pack features
    batch = collator(features)

    # Expected outputs
    expected_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    expected_labels = torch.tensor([[-100, 20, 30, -100, 50, 60, 70, -100, 90]])
    expected_position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 0, 1]])

    input_ids_correct = torch.all(batch["input_ids"] == expected_input_ids).item()
    labels_correct = torch.all(batch["labels"] == expected_labels).item()
    position_ids_correct = torch.all(batch["position_ids"] == expected_position_ids).item()

    print(f"Input IDs correct: {input_ids_correct}")
    print(f"Labels correct: {labels_correct}")
    print(f"Position IDs correct: {position_ids_correct}")

    assert input_ids_correct, "Input IDs not packed correctly"
    assert labels_correct, "Labels not packed correctly"
    assert position_ids_correct, "Position IDs not generated correctly"

    print("Collator test passed!")


def benchmark_document_masking(seq_lengths=None):
    """
    Benchmark document-aware Flash Attention with position IDs vs regular attention.

    Args:
        seq_lengths: List of sequence lengths to test. If None, use default lengths.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device = torch.device("cuda")

    # Parameters
    batch_size = 1
    hidden_size = 768
    num_heads = 12
    head_dim = hidden_size // num_heads

    if seq_lengths is None:
        # Default: 10 sequences of varying lengths
        seq_lengths = [32, 64, 128, 256, 512, 48, 96, 192, 384, 768]

    total_seq_len = sum(seq_lengths)

    print(f"Testing with {len(seq_lengths)} sequences, total length {total_seq_len}")

    # Create input tensors
    q = torch.randn(batch_size, total_seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, total_seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, total_seq_len, hidden_size, device=device, dtype=torch.bfloat16)

    # Create position IDs
    position_ids = torch.zeros((batch_size, total_seq_len), dtype=torch.long, device=device)

    # Populate position IDs
    current_pos = 0
    for seq_len in seq_lengths:
        for i in range(seq_len):
            position_ids[0, current_pos + i] = i
        current_pos += seq_len

    # Initialize attention modules
    flash_attn = DocumentAwareFlashAttention(hidden_size, num_heads).to(device)

    # Create document mask for standard attention
    document_mask = torch.zeros((batch_size, 1, total_seq_len, total_seq_len), device=device)
    start_idx = 0
    for seq_len in seq_lengths:
        document_mask[0, 0, start_idx : start_idx + seq_len, start_idx : start_idx + seq_len] = 1
        start_idx += seq_len

    # Convert to bfloat16 for Flash Attention compatibility
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    document_mask = document_mask.to(torch.bfloat16)

    # Warmup
    for _ in range(10):
        flash_attn(q, k, v, position_ids=position_ids)
        F.scaled_dot_product_attention(
            q.view(batch_size, total_seq_len, num_heads, head_dim).transpose(1, 2),
            k.view(batch_size, total_seq_len, num_heads, head_dim).transpose(1, 2),
            v.view(batch_size, total_seq_len, num_heads, head_dim).transpose(1, 2),
            attn_mask=document_mask,
        )

    # Benchmark standard attention
    torch.cuda.synchronize()
    start_time = time.time()
    num_runs = 100

    for _ in range(num_runs):
        out_standard = F.scaled_dot_product_attention(
            q.view(batch_size, total_seq_len, num_heads, head_dim).transpose(1, 2),
            k.view(batch_size, total_seq_len, num_heads, head_dim).transpose(1, 2),
            v.view(batch_size, total_seq_len, num_heads, head_dim).transpose(1, 2),
            attn_mask=document_mask,
        )

    torch.cuda.synchronize()
    standard_time = (time.time() - start_time) / num_runs

    # Benchmark flash attention
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_runs):
        out_flash = flash_attn(q, k, v, position_ids=position_ids)

    torch.cuda.synchronize()
    flash_time = (time.time() - start_time) / num_runs

    speedup = standard_time / flash_time

    print(f"Standard Attention time: {standard_time * 1000:.3f} ms")
    print(f"Flash Attention time: {flash_time * 1000:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")

    # Verify outputs are close
    out_standard = out_standard.transpose(1, 2).reshape(batch_size, total_seq_len, hidden_size)
    error = (out_standard - out_flash).abs().mean().item()
    print(f"Mean absolute error: {error:.6f}")


if __name__ == "__main__":
    print("Testing document masking with position IDs...")
    test_document_masking_with_position_ids()

    print("\nTesting position ID aware collator...")
    test_position_id_aware_collator()

    print("\nBenchmarking document masking...")
    benchmark_document_masking()

    print("\nBenchmarking with longer sequences...")
    benchmark_document_masking(seq_lengths=[512, 1024, 2048, 512, 1024, 2048])
