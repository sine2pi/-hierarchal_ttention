  
  
    class HierarchicalAttention(nn.Module):
      def __init__(self, dims, head, sharpen=True, max_dist=128, levels=3, temp_scale=0.01):
          super().__init__()
          self.dims = dims
          self.head = head
          self.max_dist = max_dist
          self.sharpen = sharpen
          self.levels = levels
          self.temp_scale = temp_scale
  
          self.span_predictor = Linear(in_features=dims, out_features=1)
  
          self.local_level_projections = nn.ModuleList([
              Linear(dims, dims) for _ in range(levels)
          ])
          self.local_level_attentions = nn.ModuleList([
              MultiheadA3(dims=dims, head=head, max_dist=max_dist) for _ in range(levels)
          ])
          self.global_level_projections = nn.ModuleList([
              Linear(dims, dims) for _ in range(levels)
          ])
          self.global_level_attentions = nn.ModuleList([
              MultiheadA3(dims=dims, head=head, max_dist=max_dist) for _ in range(levels)
          ])
  
          self.ln_local = LayerNorm(normalized_shape=dims)
          self.ln_global = LayerNorm(normalized_shape=dims)
          self.projection = Linear(in_features=2 * dims, out_features=dims)
  
      def forward(self, x):
          local = self.ln_local(x)
          global_ = self.ln_global(x)
  
          globe_out = self._hierarchical_attention(global_, self.global_level_projections, self.global_level_attentions)  # (seq_len, batch_size, dims)
          span_scale = torch.sigmoid(self.span_predictor(globe_out.mean(dim=1)))
          local_out = self._hierarchical_attention(local, self.local_level_projections, self.local_level_attentions)
          combined = torch.cat([local_out, globe_out], dim=-1)
          x = self.projection(combined)
          return x
  
      def _hierarchical_attention(self, x, level_projections, level_attentions):
          seq_len, batch_size, dims = x.size()
          outputs = []
          max_downsample_level = min(self.levels, int(math.log2(seq_len)))
          
          for level in range(max_downsample_level):
              factor = 2 ** level
              curr_len = seq_len // factor
              pooled_x = x[:curr_len * factor].view(curr_len, factor, batch_size, dims).mean(dim=1)
              
              projected = level_projections[level](pooled_x)
              attention_out, _ = level_attentions[level](projected, projected, projected)
              
              if factor > 1:
                  attention_out = F.interpolate(
                      attention_out.permute(1, 2, 0),
                      size=seq_len,
                      mode='linear',
                      align_corners=False
                  ).permute(2, 0, 1)  
              
              outputs.append(attention_out)
          
          for level in range(max_downsample_level, self.levels):
              projected = level_projections[level](x)
              attention_out, _ = level_attentions[level](projected, projected, projected)
              outputs.append(attention_out)
          
          weights = torch.softmax(torch.ones(len(outputs)), dim=0)
          combined_output = sum(out * w for out, w in zip(outputs, weights))
          
          return combined_output
