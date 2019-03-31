import torch
import torch.autograd as autograd


class SVGD:
  def __init__(self, P, K, optimizer):
    self.P = P
    self.K = K
    self.optim = optimizer

  def phi(self, X):
    X = X.detach().requires_grad_(True)

    log_prob = self.P.log_prob(X)
    score_func = autograd.grad(log_prob, X,
                               grad_outputs=torch.ones_like(log_prob),
                               only_inputs=True)[0]

    K_XX = self.K(X, X.detach())
    grad_K = -autograd.grad(K_XX, X,
                            grad_outputs=torch.ones_like(K_XX),
                            only_inputs=True)[0]

    phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

    return phi

  def step(self, X):
    self.optim.zero_grad()
    X.grad = -self.phi(X)
    self.optim.step()
