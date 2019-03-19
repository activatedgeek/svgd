import torch

class SVGD:
  def __init__(self, P, K, eta=1e-2, rho=0.9):
    self.P = P
    self.K = K
    self.eta = eta
    self.rho = rho

    self._phi_est = None
    self.reset()

  def reset(self):
    self._phi_est = None

  def step(self, X):
    X = X.detach().requires_grad_(True)

    log_prob = self.P.log_prob(X)
    score_func = torch.autograd.grad(log_prob, X,
                                     grad_outputs=torch.ones_like(log_prob),
                                     only_inputs=True)[0]

    K_XX = self.K(X, X.detach())
    grad_K = -torch.autograd.grad(K_XX, X,
                                  grad_outputs=torch.ones_like(K_XX),
                                  only_inputs=True)[0]

    phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

    if self._phi_est is None:
      self._phi_est = phi ** 2
    else:
      self._phi_est = self.rho * self._phi_est + (1 - self.rho) * phi ** 2

    grad = phi / (1e-8 + self._phi_est.sqrt())

    return X.detach() + self.eta * grad
