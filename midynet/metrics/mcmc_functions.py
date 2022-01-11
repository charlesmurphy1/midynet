# def exact_logEvidence(model, state=None, **kwargs):
#     g = model.graph.copy_state()
#     logp = []
#     for gg in model.graph.all_graphs():
#         model.set_graph(gg)
#         logp.append(-model.entropy(state=state))
#     model.set_graph(g)
#     return log_sum_exp(logp)
#
#
# def annealed_logEvidence(model, state=None, kmax=10, alpha=0.5, **kwargs):
#     kwargs.setdefault("verbose", 0)
#     logF = []
#     beta_k = (np.linspace(0, 1, kmax + 1)) ** (1.0 / alpha)
#     lower_betas = beta_k[:-1]
#     upper_betas = beta_k[1:]
#     params = model.copy_params()
#     for lb, ub in zip(lower_betas, upper_betas):
#         logr_k = []
#         queue = []
#         if kwargs["verbose"] == 1 or kwargs["verbose"] == 2:
#             print(f"Inverse temperature: {lb}")
#         mcmc_equilibriate(model, state=state, queue=queue, beta=lb, **kwargs)
#         ll = []
#         for p in queue:
#             model.set_params(p)
#             logr_k.append((ub - lb) * model.loglikelihood(state=state))
#         logF.append(log_mean_exp(logr_k))
#     model.set_params(params)
#     return np.sum(logF)

#
# def arithmetic_logEvidence(model):
#     logF = []
#     params = model.copy_params()
#     for m in range(M):
#         logF_k = []
#         for k in range(num_sweeps):
#             model.resample()
#             logF_k.append(model.loglikelihood(state))
#         logF.append(log_mean_exp(logF_k))
#     model.set_params(params)
#     return np.mean(logF)


# def harmonic_logEvidence(model, state=None, **kwargs):
#     queue = []
#     kwargs.setdefault("verbose", 0)
#     params = model.copy_params()
#     mcmc_equilibriate(model, state=state, queue=queue, **kwargs)
#     logF = []
#     for p in queue:
#         model.set_params(p)
#         logF.append(-model.loglikelihood(state))
#     model.set_params(params)
#     return -log_mean_exp(logF)
#
#
# def meanfield_logEvidence(model, state=None, **kwargs):
#     kwargs.setdefault("verbose", 0)
#     hp_x = meanfield_logPosterior(model, state=state, **kwargs)
#     hxp = -model.entropy()
#     return hxp - hp_x
#
#
# def exact_meanfield_logEvidence(model, state=None, **kwargs):
#     kwargs.setdefault("verbose", 0)
#     hp_x = exact_meanfield_logPosterior(model, state=state, **kwargs)
#     hxp = -model.entropy()
#     return hxp - hp_x
#
#
# def exact_logPosterior(model, state=None, params=None, **kwargs):
#     if params is not None:
#         model.set_params(params)
#     hx = exact_logEvidence(model, state=state, **kwargs)
#     hxp = -model.entropy(state=state)
#     return hxp - hx
#
#
# def annealed_logPosterior(model, state=None, params=None, **kwargs):
#     if params is not None:
#         model.set_params(params)
#     hx = annealed_logEvidence(model, state=state, **kwargs)
#     hxp = -model.entropy(state=state)
#     return hxp - hx
#
#
# def harmonic_logPosterior(model, state=None, params=None, **kwargs):
#     if params is not None:
#         model.set_params(params)
#     hx = harmonic_logEvidence(model, state=state, **kwargs)
#     hxp = -model.entropy(state=state)
#     return hxp - hx
#
#
# def arithmetic_logPosterior(model, state=None, params=None, **kwargs):
#     if params is not None:
#         model.set_params(params)
#     hx = arithmetic_logEvidence(model, state=state, **kwargs)
#     hxp = -model.entropy(state=state)
#     return hxp - hx
#
#
# def meanfield_logPosterior(model, state=None, params=None, **kwargs):
#     kwargs.setdefault("verbose", 0)
#     if params is not None:
#         model.set_params(params)
#     mcmc_equilibriate(model, state=state, collect=True, **kwargs)
#     hg_x = -model.graph.marginal_entropy()
#     return hg_x
#
#
# def exact_meanfield_logPosterior(model, state=None, params=None, **kwargs):
#     if params is not None:
#         model.set_params(params)
#
#     g = model.graph.copy_state()
#     edge_posterior = defaultdict(int)
#     logp = []
#     evidence = exact_logEvidence(model, state=state)
#     for gg in model.graph.all_graphs():
#         model.set_graph(gg)
#         h = -model.entropy(state=state) - evidence
#         logp.append(h)
#         for e in gg.edges():
#             i, j = sort_index(int(e.source()), int(e.target()))
#             edge_posterior[i, j] += np.exp(h)
#     model.set_graph(g)
#
#     hg_x = 0
#     for (i, j), pp in edge_posterior.items():
#         if g.edge(i, j) is not None:
#             hg_x += np.log(pp)
#         else:
#             hg_x += np.log(1 - pp)
#     return hg_x
