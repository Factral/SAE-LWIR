from framework.modules import *

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


class RowWiseFF(nn.Module):
    """
    Applies the same MLP to each element in the set (last-dim features).
    Input:  (B, K, Din)
    Output: (B, K, Dout)
    """

    def __init__(self, dim_in, dim_hidden, dim_out, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.relu1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.relu2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.ln2 = nn.LayerNorm(dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x1 = self.ln1(x)
        x = self.fc2(x1)
        x = self.relu2(x)   
        x = self.drop2(x)
        x = self.ln2(x)
        x = self.fc3(x)
        return x


class PerRowFF(nn.Module):
    """
    Like RowWiseFF, but with different parameters per row index.

    Input:  (B, K, Din)
    Output: (B, K, Dout)
    """

    def __init__(self, set_size, dim_in, dim_hidden, dim_out, dropout=0.0):
        super().__init__()
        self.set_size = int(set_size)
        self.rows = nn.ModuleList(
            [RowWiseFF(dim_in, dim_hidden, dim_out, dropout=dropout) for _ in range(self.set_size)]
        )

    def forward(self, x):
        if x.size(1) != self.set_size:
            raise ValueError(f"Expected set_size={self.set_size}, got x.shape={tuple(x.shape)}")
        outs = []
        for i, mlp in enumerate(self.rows):
            outs.append(mlp(x[:, i : i + 1, :]))  # (B,1,Dout)
        return torch.cat(outs, dim=1)  # (B,K,Dout)


class MultiHeadSetTransformer(nn.Module):
    """
    Encoder: ISAB -> ISAB
    Decoder: three branches
      - T branch (transmittance): RowWiseFF + sigmoid
      - U branch (upwelling):     RowWiseFF
      - D branch (downwelling):   PMA -> SAB -> Linear

    Shapes (matching current set_transformer/data.py):
      x:  (B, 7, 256)
      y1: (B, 7, 256)  (T)
      y2: (B, 7, 256)  (U)
      y3: (B, 1, 256)  (D)
    """

    def __init__(
        self,
        dim_input=256,
        set_size=7,
        dim_hidden=256,
        num_heads=4,
        num_inds=32,
        ln=True,
        dropout=0.1,
        rowwise=True
    ):
        super().__init__()

        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )

        # T (transmittance): constrain to [0,1] via sigmoid
        if rowwise:
            self.dec_t = RowWiseFF(dim_hidden, dim_hidden, dim_input, dropout=dropout)
        else:
            self.dec_t = PerRowFF(set_size, dim_hidden, dim_hidden, dim_input, dropout=dropout)

        # U (upwelling)
        if rowwise:
            self.dec_u = RowWiseFF(dim_hidden, dim_hidden, dim_input, dropout=dropout)
        else:
            self.dec_u = PerRowFF(set_size, dim_hidden, dim_hidden, dim_input, dropout=dropout)

        # D (downwelling): a single global prediction per sample
        self.dec_d = nn.Sequential(
            PMA(dim_hidden, num_heads, 1, ln=ln),
            nn.Linear(dim_hidden, dim_input),
        )


    def encode(self, x):
        return self.enc(x)  # (B,K,H)


    def decode(self, h):
        t = torch.sigmoid(self.dec_t(h))
        u = torch.relu(self.dec_u(h))
        d = self.dec_d(h)
        return t, u, d


    def forward(self, x, *, return_h=False, h_override=None):
        h = self.encode(x) if h_override is None else h_override
        t, u, d = self.decode(h)
        if return_h:
            return t, u, d, h
        return t, u, d