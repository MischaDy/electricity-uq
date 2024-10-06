import pickle

import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import torch

from laplace import Laplace, marglik_training, LLLaplace

from helpers import IO_Helper


import warnings

warnings.simplefilter("ignore")
# todo: why are there warnings?


N_POINTS_TEMP = 100  # per group

IO_HELPER = IO_Helper("laplace_storage")

n_epochs = 100
batch_size = 1
torch.manual_seed(711)

torch.set_default_dtype(torch.float64)


_data_pickle_path = '_laplace_data.pkl'


# with open(_data_pickle_path, 'wb') as file:
#     pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


with open(_data_pickle_path, 'rb') as file:
    data = (X_train, X_test, y_train, y_test) = pickle.load(file)


train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size)


_MODEL_PATH = "_laplace_base.pth"
# torch.save(model, temp_path)

model = torch.load(_MODEL_PATH, weights_only=False)
model.eval()



# ### Fitting and optimizing the Laplace approximation using empirical Bayes

# With the MAP-trained model at hand, we can estimate the prior precision and observation noise
# using empirical Bayes after training.
# We fit the LA to the training data and initialize `log_prior` and `log_sigma`.
# Using Adam, we minimize the negative log marginal likelihood for `n_epochs`.



from timeit import default_timer

# noinspection PyTypeChecker
la: LLLaplace = Laplace(
    model, "regression", subset_of_weights="last_layer", hessian_structure="kron"
)
print(la.sigma_noise.item())
t1 = default_timer()
la.fit(train_loader, progress_bar=True)
t2 = default_timer()
print(f"time for fitting: {round(t2-t1)}s")  # ~140s
print(la.sigma_noise.item())


_fitted_la_path = '_fitted_la_state_dict.pkl'


# with open(_fitted_la_path, 'wb') as file:
#     pickle.dump(la.state_dict(), file, pickle.HIGHEST_PROTOCOL)


with open(_fitted_la_path, 'rb') as file:
    state_dict = pickle.load(file)

la.load_state_dict(state_dict)


# ----

f_mu, f_var = la(X_test)
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_sigma = f_var.squeeze().sqrt().cpu().numpy()
pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)


# ------------

log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(
    1, requires_grad=True
)
hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)

n_epochs_lap = 1000
for epoch in range(n_epochs_lap):
    if epoch % (n_epochs_lap // 10) == 0:
        print("epoch:", epoch)
    hyper_optimizer.zero_grad()
    neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()
print("done!")
print(la.sigma_noise.item())


# ### Bayesian predictive
#
# Here, we compare the MAP prediction to the obtained LA prediction.
# For LA, we have a closed-form predictive distribution on the output \\(f\\) which is a Gaussian
# \\(\mathcal{N}(f(x;\theta\_{MAP}), \mathbb{V}[f] + \sigma^2)\\):


f_mu, f_var = la(X_test)
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_sigma = f_var.squeeze().sqrt().cpu().numpy()
pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)


# In comparison to the MAP, the predictive shows useful uncertainties.
# When our MAP is over or underfit, the Laplace approximation cannot fix this anymore.
# In this case, joint optimization of MAP and marginal likelihood can be useful.

# ### Jointly optimize MAP and hyperparameters using online empirical Bayes
#
# We provide a utility method `marglik_training` that implements the algorithm proposed in [1].
# The method optimizes the neural network and the hyperparameters in an interleaved way
# and returns an optimally regularized LA.
# Below, we use this method and plot the corresponding predictive uncertainties again:


from laplace.curvature.backpack import BackPackGGN


model = torch.load(_MODEL_PATH, weights_only=False)
la, model, margliks, losses = marglik_training(
    model=model,
    train_loader=train_loader,
    likelihood="regression",
    # hessian_structure="full",
    backend=BackPackGGN,
    n_epochs=n_epochs,
    optimizer_kwargs={"lr": 1e-3},
    prior_structure="scalar",
)


f_mu, f_var = la(X_test)
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_sigma = f_var.squeeze().sqrt().cpu().numpy()
pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)


print(la.sigma_noise.item())
