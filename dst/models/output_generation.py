import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np

class SparseAttention(nn.Module):
    """Sparse attention mechanism for efficient sequence processing"""
    
    def __init__(self, d_model: int = 768, num_heads: int = 12, window_size: int = 512, 
                 top_k: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.top_k = top_k
        self.head_dim = d_model // num_heads
        
        # Standard attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def windowed_attention(self, query: torch.Tensor, key: torch.Tensor, 
                          value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Local windowed attention for nearby tokens"""
        batch_size, seq_len, _ = query.shape
        
        # Reshape for multi-head attention
        q = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Create windowed attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply windowed mask
        if seq_len > self.window_size:
            window_mask = self._create_window_mask(seq_len, self.window_size, query.device)
            attention_scores = attention_scores.masked_fill(window_mask == 0, float('-inf'))
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return context
    
    def sparse_top_k_attention(self, query: torch.Tensor, key: torch.Tensor, 
                              value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sparse top-k attention for global context"""
        batch_size, seq_len, _ = query.shape
        
        # Reshape for multi-head attention
        q = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply top-k sparsity
        if seq_len > self.top_k:
            # Keep only top-k scores for each query
            top_k_scores, top_k_indices = torch.topk(attention_scores, 
                                                    min(self.top_k, seq_len), dim=-1)
            
            # Create sparse attention matrix
            sparse_scores = torch.full_like(attention_scores, float('-inf'))
            sparse_scores.scatter_(-1, top_k_indices, top_k_scores)
            attention_scores = sparse_scores
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return context
    
    def _create_window_mask(self, seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """Create causal windowed attention mask"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask * causal_mask
        
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass combining windowed and sparse attention
        """
        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Apply windowed attention for local context
        windowed_output = self.windowed_attention(q, k, v, mask)
        
        # Apply sparse attention for global context
        sparse_output = self.sparse_top_k_attention(q, k, v, mask)
        
        # Combine local and global attention
        combined_output = (windowed_output + sparse_output) / 2.0
        
        # Final projection
        output = self.out_proj(combined_output)
        
        return output

class SharedDecoderLayer(nn.Module):
    """Shared transformer decoder layer with sparse attention"""
    
    def __init__(self, d_model: int = 768, num_heads: int = 12, d_ff: int = 3072, 
                 dropout: float = 0.1, window_size: int = 512, top_k: int = 64):
        super().__init__()
        self.d_model = d_model
        
        # Sparse multi-head attention
        self.self_attention = SparseAttention(d_model, num_heads, window_size, top_k, dropout)
        
        # Cross-attention for encoder-decoder attention
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: Optional[torch.Tensor] = None,
                self_attn_mask: Optional[torch.Tensor] = None, 
                cross_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with self-attention and optional cross-attention
        """
        # Self-attention with residual connection
        attn_output = self.self_attention(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention (if encoder output provided)
        if encoder_output is not None:
            cross_attn_output, _ = self.cross_attention(
                query=x, key=encoder_output, value=encoder_output, attn_mask=cross_attn_mask
            )
            x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x

class SharedDecoderBackbone(nn.Module):
    """Shared transformer decoder backbone for efficient dual-head processing"""
    
    def __init__(self, d_model: int = 768, num_layers: int = 6, num_heads: int = 12,
                 d_ff: int = 3072, dropout: float = 0.1, window_size: int = 512, top_k: int = 64):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Shared decoder layers
        self.layers = nn.ModuleList([
            SharedDecoderLayer(d_model, num_heads, d_ff, dropout, window_size, top_k)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, d_model))
        
    def forward(self, x: torch.Tensor, encoder_output: Optional[torch.Tensor] = None,
                self_attn_mask: Optional[torch.Tensor] = None, 
                cross_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through shared decoder layers
        """
        batch_size, seq_len, _ = x.shape
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Pass through shared decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x

class DetailGenerationHead(nn.Module):
    """Detail generation head for low-level JSON output"""
    
    def __init__(self, d_model: int = 768, vocab_size: int = 50000, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Task-specific layers
        self.detail_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # JSON syntax embedding for constrained generation
        self.json_tokens = nn.Parameter(torch.randn(100, d_model))  # Special JSON tokens
        
    def forward(self, shared_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate detailed JSON tokens from shared features
        """
        # Transform shared features for detail generation
        detail_features = self.detail_transform(shared_features)
        
        # Project to vocabulary space
        detail_logits = self.output_projection(detail_features)
        
        # JSON syntax guidance (simplified)
        json_guidance = torch.matmul(detail_features, self.json_tokens.transpose(0, 1))
        
        return {
            'detail_logits': detail_logits,
            'json_guidance': json_guidance,
            'detail_features': detail_features
        }

class SemanticGenerationHead(nn.Module):
    """Semantic generation head for high-level layout elements"""
    
    def __init__(self, d_model: int = 768, num_semantic_classes: int = 50):
        super().__init__()
        self.d_model = d_model
        self.num_semantic_classes = num_semantic_classes
        
        # Semantic-specific transformation
        self.semantic_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Semantic classification layers
        self.semantic_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_semantic_classes)
        )
        
        # Semantic element embeddings
        self.semantic_embeddings = nn.Parameter(torch.randn(num_semantic_classes, d_model))
        
    def forward(self, shared_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate semantic layout elements from shared features
        """
        # Transform shared features for semantic understanding
        semantic_features = self.semantic_transform(shared_features)
        
        # Semantic classification
        semantic_logits = self.semantic_classifier(semantic_features)
        
        # Semantic similarity scores
        semantic_similarities = torch.matmul(semantic_features, self.semantic_embeddings.transpose(0, 1))
        
        return {
            'semantic_logits': semantic_logits,
            'semantic_similarities': semantic_similarities,
            'semantic_features': semantic_features
        }

class ElementTypeClassifier(nn.Module):
    """Multi-class element type classifier for parallel supervision"""
    
    def __init__(self, d_model: int = 768, num_element_types: int = 30):
        super().__init__()
        self.d_model = d_model
        self.num_element_types = num_element_types
        
        # Attention pooling for sequence-level features
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True
        )
        
        # Element type classification layers
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, num_element_types)
        )
        
        # Element type embeddings
        self.element_embeddings = nn.Parameter(torch.randn(num_element_types, d_model))
        
    def forward(self, shared_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Classify element types from shared decoder features
        """
        # Attention pooling for global features
        pooled_features, attention_weights = self.attention_pooling(
            shared_features, shared_features, shared_features
        )
        
        # Global pooling for sequence-level classification
        global_features = pooled_features.mean(dim=1)  # [batch, d_model]
        
        # Element type classification
        element_logits = self.classifier(global_features)
        
        # Element similarity scores
        element_similarities = torch.matmul(global_features, self.element_embeddings.transpose(0, 1))
        
        return {
            'element_logits': element_logits,
            'element_similarities': element_similarities,
            'pooled_features': pooled_features,
            'attention_weights': attention_weights
        }

class MultiTaskLossFunction(nn.Module):
    """Multi-task loss function for integrated training"""
    
    def __init__(self, 
                 lambda_detail: float = 0.4,
                 lambda_semantic: float = 0.3,
                 lambda_element: float = 0.2,
                 lambda_consistency: float = 0.1):
        super().__init__()
        self.lambda_detail = lambda_detail
        self.lambda_semantic = lambda_semantic
        self.lambda_element = lambda_element
        self.lambda_consistency = lambda_consistency
        
        # Loss functions
        self.detail_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.semantic_loss = nn.CrossEntropyLoss()
        self.element_loss = nn.BCEWithLogitsLoss()
        self.consistency_loss = nn.MSELoss()
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss with adaptive weighting
        """
        losses = {}
        
        # Detail generation loss
        if 'detail_logits' in predictions and 'detail_targets' in targets:
            detail_logits = predictions['detail_logits']
            detail_targets = targets['detail_targets']
            
            # Reshape for sequence loss
            if detail_logits.dim() == 3:
                detail_logits = detail_logits.view(-1, detail_logits.size(-1))
                detail_targets = detail_targets.view(-1)
            
            losses['detail_loss'] = self.detail_loss(detail_logits, detail_targets)
        
        # Semantic generation loss
        if 'semantic_logits' in predictions and 'semantic_targets' in targets:
            semantic_logits = predictions['semantic_logits']
            semantic_targets = targets['semantic_targets']
            
            # Handle sequence or classification format
            if semantic_logits.dim() == 3:
                semantic_logits = semantic_logits.view(-1, semantic_logits.size(-1))
                semantic_targets = semantic_targets.view(-1)
            
            losses['semantic_loss'] = self.semantic_loss(semantic_logits, semantic_targets)
        
        # Element type classification loss
        if 'element_logits' in predictions and 'element_targets' in targets:
            losses['element_loss'] = self.element_loss(
                predictions['element_logits'], 
                targets['element_targets']
            )
        
        # Consistency loss between detail and semantic features
        if 'detail_features' in predictions and 'semantic_features' in predictions:
            detail_features = predictions['detail_features'].mean(dim=1)
            semantic_features = predictions['semantic_features'].mean(dim=1)
            losses['consistency_loss'] = self.consistency_loss(detail_features, semantic_features)
        
        # Compute total loss
        total_loss = 0.0
        if 'detail_loss' in losses:
            total_loss += self.lambda_detail * losses['detail_loss']
        if 'semantic_loss' in losses:
            total_loss += self.lambda_semantic * losses['semantic_loss']
        if 'element_loss' in losses:
            total_loss += self.lambda_element * losses['element_loss']
        if 'consistency_loss' in losses:
            total_loss += self.lambda_consistency * losses['consistency_loss']
        
        losses['total_loss'] = total_loss
        
        return losses

class OutputGenerationModule(nn.Module):
    """Complete output generation module with dual-headed architecture"""
    
    def __init__(self, 
                 d_model: int = 768, 
                 num_decoder_layers: int = 6,
                 num_heads: int = 12,
                 vocab_size: int = 50000,
                 num_semantic_classes: int = 50,
                 num_element_types: int = 30,
                 window_size: int = 512,
                 top_k: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_semantic_classes = num_semantic_classes
        self.num_element_types = num_element_types
        
        # Shared decoder backbone
        self.shared_decoder = SharedDecoderBackbone(
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            window_size=window_size,
            top_k=top_k,
            dropout=dropout
        )
        
        # Dual generation heads
        self.detail_head = DetailGenerationHead(d_model, vocab_size)
        self.semantic_head = SemanticGenerationHead(d_model, num_semantic_classes)
        
        # Element type classifier
        self.element_classifier = ElementTypeClassifier(d_model, num_element_types)
        
        # Input projection for fused features
        self.input_projection = nn.Linear(d_model, d_model)
        
        # Target embedding layer to convert token indices to embeddings
        self.target_embedding = nn.Embedding(vocab_size, d_model)
        
        # Special tokens
        self.bos_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.eos_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, 
                fused_features: torch.Tensor,
                target_sequence: Optional[torch.Tensor] = None,
                inference_mode: str = 'dual') -> Dict[str, torch.Tensor]:
        """
        Generate structured JSON output from multimodal fusion features
        
        Args:
            fused_features: [batch, seq_len, d_model] from multimodal fusion
            target_sequence: [batch, target_seq_len] token indices for training
            inference_mode: 'dual', 'detail', or 'semantic'
        """
        batch_size = fused_features.size(0)
        
        # Project input features
        encoder_output = self.input_projection(fused_features)
        
        # Prepare decoder input
        if target_sequence is not None:
            # Training mode - teacher forcing
            # Convert token indices to embeddings
            decoder_input = self.target_embedding(target_sequence)
        else:
            # Inference mode - start with BOS token
            decoder_input = self.bos_token.expand(batch_size, 1, -1)
        
        # Pass through shared decoder
        shared_features = self.shared_decoder(
            x=decoder_input,
            encoder_output=encoder_output
        )
        
        # Generate outputs based on mode
        outputs = {'shared_features': shared_features}
        
        if inference_mode in ['dual', 'detail']:
            detail_outputs = self.detail_head(shared_features)
            outputs.update(detail_outputs)
        
        if inference_mode in ['dual', 'semantic']:
            semantic_outputs = self.semantic_head(shared_features)
            outputs.update(semantic_outputs)
        
        # Element type classification (always computed)
        element_outputs = self.element_classifier(shared_features)
        outputs.update(element_outputs)
        
        return outputs
    
    def generate_autoregressive(self, 
                              fused_features: torch.Tensor,
                              max_length: int = 512,
                              temperature: float = 1.0,
                              top_k: int = 50,
                              top_p: float = 0.9) -> Dict[str, torch.Tensor]:
        """
        Autoregressive generation for inference
        """
        batch_size = fused_features.size(0)
        device = fused_features.device
        
        # Initialize generation
        generated_tokens = []
        current_input = self.bos_token.expand(batch_size, 1, -1)
        
        encoder_output = self.input_projection(fused_features)
        
        for step in range(max_length):
            # Forward pass
            shared_features = self.shared_decoder(
                x=current_input,
                encoder_output=encoder_output
            )
            
            # Get detail logits for next token
            detail_outputs = self.detail_head(shared_features)
            next_token_logits = detail_outputs['detail_logits'][:, -1, :]  # Last token
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k and top-p sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated_tokens.append(next_token)
            
            # Update input for next iteration
            next_token_embedding = self.detail_head.detail_transform(
                F.embedding(next_token, self.detail_head.output_projection.weight.transpose(0, 1))
            )
            current_input = torch.cat([current_input, next_token_embedding], dim=1)
            
            # Check for EOS token (assuming token 2 is EOS)
            if (next_token == 2).all():
                break
        
        return {
            'generated_tokens': torch.cat(generated_tokens, dim=1),
            'generation_length': len(generated_tokens)
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize the output generation module
    output_module = OutputGenerationModule(
        d_model=768,
        num_decoder_layers=6,
        num_heads=12,
        vocab_size=50000,
        num_semantic_classes=50,
        num_element_types=30,
        window_size=512,
        top_k=64
    )
    
    # Initialize multi-task loss function
    loss_function = MultiTaskLossFunction()
    
    # Test forward pass
    batch_size = 2
    seq_len = 100
    fused_features = torch.randn(batch_size, seq_len, 768)
    target_sequence = torch.randint(0, 50000, (batch_size, 50)) # Changed to token indices
    
    print("Testing Output Generation Module:")
    print(f"Input fused_features shape: {fused_features.shape}")
    print(f"Target sequence shape: {target_sequence.shape}")
    
    # Forward pass
    outputs = output_module(fused_features, target_sequence, inference_mode='dual')
    
    # Print output shapes
    print("\nOutput Generation Results:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test loss computation
    targets = {
        'detail_targets': torch.randint(0, 50000, (batch_size, 50)),
        'semantic_targets': torch.randint(0, 50, (batch_size, 50)),
        'element_targets': torch.randn(batch_size, 30)
    }
    
    # Prepare predictions for loss computation
    predictions = {
        'detail_logits': outputs['detail_logits'],
        'semantic_logits': outputs['semantic_logits'],
        'element_logits': outputs['element_logits'],
        'detail_features': outputs['detail_features'],
        'semantic_features': outputs['semantic_features']
    }
    
    losses = loss_function(predictions, targets)
    
    print("\nLoss Components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test autoregressive generation
    print("\nTesting Autoregressive Generation:")
    generation_results = output_module.generate_autoregressive(
        fused_features, max_length=20, temperature=1.0
    )
    
    print(f"Generated tokens shape: {generation_results['generated_tokens'].shape}")
    print(f"Generation length: {generation_results['generation_length']}")
    
    # Model parameters
    total_params = sum(p.numel() for p in output_module.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test sparse attention efficiency
    print("\nTesting Sparse Attention:")
    sparse_attn = SparseAttention(d_model=768, num_heads=12, window_size=256, top_k=32)
    test_input = torch.randn(2, 1000, 768)  # Long sequence
    
    # Time the forward pass
    import time
    start_time = time.time()
    sparse_output = sparse_attn(test_input, test_input, test_input)
    end_time = time.time()
    
    print(f"Sparse attention output shape: {sparse_output.shape}")
    print(f"Processing time for 1000-token sequence: {end_time - start_time:.4f} seconds") 