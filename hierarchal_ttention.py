

class FocusA(nn.Module):
    def __init__(self, base, dims, head, sharpen=True, max_dist=128, levels=3, temp_scale=0.01):
        super().__init__()
        self.base = base
        self.dims = dims
        self.head = head
        self.max_dist = max_dist
        self.sharpen = sharpen
        self.levels = levels
        self.temp_scale = temp_scale

        # Span predictor
        self.span_predictor = Linear(in_features=dims, out_features=1)

        # Hierarchical attention layers for local and global contexts
        self.local_level_projections = nn.ModuleList([
            Linear(dims, dims) for _ in range(levels)
        ])
        self.local_level_attentions = nn.ModuleList([
            MultiheadA3(base=base, dims=dims, head=head, max_dist=max_dist) for _ in range(levels)
        ])
        self.global_level_projections = nn.ModuleList([
            Linear(dims, dims) for _ in range(levels)
        ])
        self.global_level_attentions = nn.ModuleList([
            MultiheadA3(base=base, dims=dims, head=head, max_dist=max_dist) for _ in range(levels)
        ])

        # Layer norms and projection
        self.ln_local = LayerNorm(normalized_shape=dims)
        self.ln_global = LayerNorm(normalized_shape=dims)
        self.projection = Linear(in_features=2 * dims, out_features=dims)

    def forward(self, x):
        # Apply layer norms
        local = self.ln_local(x)
        global_ = self.ln_global(x)

        # Global hierarchical attention
        globe_out = self._hierarchical_attention(global_, self.global_level_projections, self.global_level_attentions)  # (seq_len, batch_size, dims)

        # Predict span scale
        span_scale = torch.sigmoid(self.span_predictor(globe_out.mean(dim=1)))

        # Local hierarchical attention
        local_out = self._hierarchical_attention(local, self.local_level_projections, self.local_level_attentions)

        # Combine local and global outputs
        combined = torch.cat([local_out, globe_out], dim=-1)
        x = self.projection(combined)

        return x

    def _hierarchical_attention(self, x, level_projections, level_attentions):
        seq_len, batch_size, dims = x.size()
        outputs = []
        
        for level in range(self.levels):
            # Downsample the sequence for lower levels in the hierarchy
            factor = 2 ** level
            if factor >= seq_len:
                pooled_x = x
            else:
                pooled_x = x[:, ::factor, :]
            
            # Apply attention at the current level
            query = level_projections[level](pooled_x)
            key = level_projections[level](pooled_x)
            value = level_projections[level](pooled_x)
            attention_out, _ = level_attentions[level](query, key, value)
            
            # Upsample the attention output to the original sequence length
            if factor >= seq_len:
                upsampled_attention_out = attention_out
            else:
                upsampled_attention_out = torch.zeros((seq_len, batch_size, dims), device=x.device)
                upsampled_attention_out[:, ::factor, :] = attention_out
            
            outputs.append(upsampled_attention_out)
        
        # Combine the outputs from all levels
        combined_output = torch.stack(outputs, dim=0).mean(dim=0)
        
        return combined_output
