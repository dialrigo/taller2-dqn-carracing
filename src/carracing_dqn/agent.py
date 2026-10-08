"""Agente DQN / Double DQN: política ε-greedy, red objetivo y actualización de Bellman."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import RedQ


def dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AgenteDQN:
    def __init__(self, cfg: dict, n_acciones: int = 5, canales: int = 4, semilla: int = 0):
        self.cfg, self.n_acciones, self.dev = cfg, n_acciones, dispositivo()
        self.q = RedQ(canales, n_acciones).to(self.dev)
        self.q_obj = RedQ(canales, n_acciones).to(self.dev)
        self.q_obj.load_state_dict(self.q.state_dict()); self.q_obj.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])
        self.rng = np.random.default_rng(semilla)

    def epsilon(self, paso: int) -> float:
        f = min(paso / self.cfg["pasos_decaimiento_epsilon"], 1.0)
        return self.cfg["epsilon_inicial"] + f * (self.cfg["epsilon_final"] - self.cfg["epsilon_inicial"])

    @torch.no_grad()
    def actuar(self, obs, eps: float = 0.0) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_acciones))
        x = torch.as_tensor(np.asarray(obs), device=self.dev).unsqueeze(0)
        return int(self.q(x).argmax(1).item())

    @torch.no_grad()
    def valores_q(self, obs) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs), device=self.dev).unsqueeze(0)
        return self.q(x).squeeze(0).cpu().numpy()

    def actualizar(self, lote) -> dict:
        s, a, r, s2, fin = (t.to(self.dev) for t in lote)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.cfg["double_dqn"]:
                a2 = self.q(s2).argmax(1, keepdim=True)                 # la red en línea elige
                q_sig = self.q_obj(s2).gather(1, a2).squeeze(1)         # la red objetivo evalúa
            else:
                q_sig = self.q_obj(s2).max(1).values
            y = r + self.cfg["gamma"] * (1.0 - fin) * q_sig              # objetivo de Bellman
        perdida = F.smooth_l1_loss(q_sa, y)                              # Huber
        self.opt.zero_grad(set_to_none=True)
        perdida.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg["recorte_gradiente"])
        self.opt.step()
        return {"perdida": float(perdida.item()), "q_medio": float(q_sa.mean().item())}

    def sincronizar(self):
        self.q_obj.load_state_dict(self.q.state_dict())

    def guardar(self, ruta, extra: dict | None = None):
        torch.save({"red": self.q.state_dict(), **(extra or {})}, ruta)

    def cargar(self, ruta):
        ck = torch.load(ruta, map_location=self.dev)
        self.q.load_state_dict(ck["red"]); self.q.eval(); self.sincronizar()
        return ck
