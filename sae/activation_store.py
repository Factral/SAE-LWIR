import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm


class ActivationsStoreST:
    def __init__(self, model, loader, cfg):
        self.model = model.eval()
        self.loader = loader
        self.cfg = cfg
        self.device = cfg["device"]
        self.it = iter(loader)
        self.buffer = None
        self.buf_it = None

    @torch.no_grad()
    def _next_acts_chunk(self):
        try:
            batch = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            batch = next(self.it)

        x, y1, y2, y3, temp = batch
        x = x.to(self.device, non_blocking=True)

        _, _, _, h = self.model(x, return_h=True)         # (B,K,H)
        acts = h.reshape(-1, h.size(-1)).contiguous()     # (B*K, H)
        return acts

    @torch.no_grad()
    def _fill_buffer(self):
        chunks = [self._next_acts_chunk() for _ in range(self.cfg["num_batches_in_buffer"])]
        return torch.cat(chunks, dim=0)  # (Nbuf_tokens, H)

    def _get_dataloader(self):
        ds = torch.utils.data.TensorDataset(self.buffer)
        return torch.utils.data.DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=True)

    def next_batch(self):
        try:
            return next(self.buf_it)[0]
        except Exception:
            self.buffer = self._fill_buffer()
            dl = self._get_dataloader()
            self.buf_it = iter(dl)
            return next(self.buf_it)[0]
