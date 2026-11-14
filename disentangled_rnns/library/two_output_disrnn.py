# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Disentangled RNN and plotting functions for choice and confidence data."""

from collections.abc import Callable
import copy
import dataclasses
from typing import Optional

from disentangled_rnns.library import disrnn, multisubject_disrnn, plotting, rnn_utils
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

# Fontsizes and formatting for plots.
small = 15
medium = 18
large = 20
mpl.rcParams['grid.color'] = 'none'
mpl.rcParams['axes.facecolor'] = 'white'
plt.rcParams['svg.fonttype'] = 'none'

@dataclasses.dataclass
class DisRnnWTwoOutputConfig(disrnn.DisRnnConfig):
  """Specifies configuration for a Disentangled RNN with confidencey prediction.

  Attributes:
    confidence_net_n_units_per_layer: Number of units per layer in the
      confidencey network
    confidence_net_n_layers: Number of layers in the confidence net
      targets bottlenecks
  """

  confidence_net_n_units_per_layer: int = 8
  confidence_net_n_layers: int = 2
  confidence_net_latent_penalty: float = 1.0


class HkTwoOutputDisentangledRNN(disrnn.HkDisentangledRNN):
  """Disentangled RNN with an extra target for confidence.

  DisRNN typically have the choice prediction as the output. This class adds an
  extra output which predicts the confidence. Currently the confidence
  is expected to be a categorical value.
  """

  def __init__(
      self,
      config: DisRnnWTwoOutputConfig,
  ):
    # We are subclassing the classic DisRNN for this model. Classic disRNN is
    # used as is for the choice prediction.
    # On top of those latents we add new nets to predict the confidence.
    #if config.obs_size != 2:
    #  raise ValueError('Only implemented for observation size of 2')

    super().__init__(config)
    self._confidence_net_n_units_per_layer = (
        config.confidence_net_n_units_per_layer
    )
    self._confidence_net_n_layers = config.confidence_net_n_layers
    self._confidence_net_latent_penalty = (
        config.confidence_net_latent_penalty
    )
    self._num_predictions = config.output_size# + 2 # +2 for categorical confidence
    self._get_haiku_parameters()

  def _build_two_output_bottlenecks(self):
    """Initializes parameters for the choice/confidence net."""
    input_size = self._latent_size# + 2
    self._confidence_net_sigmas, self._confidence_net_multipliers = (
        disrnn.get_initial_bottleneck_params(
            shape=(input_size,),
            name='confidence_net',
        )
    )

  def _get_haiku_parameters(self):
    """Initializes all haiku parameters for this network."""
    super()._get_haiku_parameters()
    self._build_two_output_bottlenecks()

  def __call__(self, observations: jnp.ndarray, prev_latents: jnp.ndarray):
    # Initial penalty values. Shape is (batch_size,)
    batch_size = observations.shape[0]
    penalty = jnp.zeros(shape=(batch_size,))

    #curr_choices = observations[:, 0]
    #curr_confidences = observations[:, 1]
    # Convention: First two columns are left and right input.
    # Right now we don't have anything else in the observation but this allows
    # us to add more easily.
    #observations = observations[:, 0:2]
    choice_disrnn_output, new_latents = super().__call__(
        observations, prev_latents
    )

    choice_prediction = choice_disrnn_output[:, :-1]
    penalty += choice_disrnn_output[:, -1]

    # Confidence net
    confidence_net_inputs, confidence_net_kl = (
        disrnn.information_bottleneck(
            inputs=new_latents,
            sigmas=self._confidence_net_sigmas,
            multipliers=self._confidence_net_multipliers,
            noiseless_mode=self._noiseless_mode,
        )
    )
    penalty += self._confidence_net_latent_penalty * confidence_net_kl

    confidence_prediction, confidence_net_l2 = disrnn.ResMLP(
        input_size=confidence_net_inputs.shape[1],
        output_size=2,         # categorical output: very/not confident
        n_units_per_layer=self._confidence_net_n_units_per_layer,
        n_layers=self._confidence_net_n_layers,
        activation_fn=self._activation,
        name='confidence_net',
    )(confidence_net_inputs)

    if not self._noiseless_mode:
      penalty += self._l2_scale * confidence_net_l2

    # Note that the final output is the choice and confidence prediction.
    output_shape = (batch_size, self._num_predictions + 2 + 1) # confidence (2), penalty (1)
    output = jnp.zeros(output_shape)
    output = output.at[:, 0:2].set(choice_prediction)
    output = output.at[:, 2:4].set(confidence_prediction)
    output = output.at[:, -1].set(penalty)

    return output, new_latents

def plot_bottlenecks(
    params: hk.Params,
    disrnn_config: DisRnnWTwoOutputConfig,
    sort_latents: bool = True,
) -> plt.Figure:
  """Plot the bottleneck sigmas from an hk.TwoOututDisentangledRNN."""

  params_disrnn = params['hk_two_output_disentangled_rnn']
  subject_embedding_size = 0
  update_input_names = disrnn_config.x_names
  # For update_sigmas: concatenate transposed reparameterized sigmas
  # Order of inputs to update nets: observations, latents
  update_obs_sigmas_t = np.transpose(
    disrnn.reparameterize_sigma(
      params_disrnn['update_net_obs_sigma_params']
    )
  )
  update_latent_sigmas_t = np.transpose(
    disrnn.reparameterize_sigma(
      params_disrnn['update_net_latent_sigma_params']
    )
  )
  update_sigmas = np.concatenate(
    (update_obs_sigmas_t, update_latent_sigmas_t), axis=1)
  choice_sigmas = np.array(
    disrnn.reparameterize_sigma(
      np.transpose(params_disrnn['choice_net_sigma_params'])
    )
  )
  confidence_sigmas = np.array(
    disrnn.reparameterize_sigma(
      np.transpose(params_disrnn['confidence_net_sigma_params'])
    )
  )
  latent_sigmas = np.array(
    disrnn.reparameterize_sigma(
      params_disrnn['latent_sigma_params']
    )
  )

  if sort_latents:
    latent_sigma_order = np.argsort(latent_sigmas)
    latent_sigmas = latent_sigmas[latent_sigma_order]

    # Sort choice sigmas based on the order of latents, keeping subject
    # embedding dimensions first if they exist.
    choice_sigma_order = np.concatenate(
      (
        np.arange(0, subject_embedding_size),
        subject_embedding_size + latent_sigma_order,
      ),
      axis=0,
    )
    choice_sigmas = choice_sigmas[choice_sigma_order]
    confidence_sigmas = confidence_sigmas[choice_sigma_order]

    # Sort update sigmas based on the order of latents, keeping subject
    # embedding dimensions first if they exist, then observations, then latents.
    non_latent_input_size = subject_embedding_size + disrnn_config.obs_size
    update_sigma_order = np.concatenate(
      (
        np.arange(0, non_latent_input_size, 1),
        non_latent_input_size + latent_sigma_order,
      ),
      axis=0,
    )
    update_sigmas = update_sigmas[latent_sigma_order, :]
    update_sigmas = update_sigmas[:, update_sigma_order]

  latent_names = np.arange(1, disrnn_config.latent_size + 1)
  fig, axes = plt.subplots(1, 4, figsize=(20, 5))

  # Plot Latent Bottlenecks on axes[0]
  im1 = axes[0].imshow(np.swapaxes([1 - latent_sigmas], 0, 1), cmap='Oranges')
  im1.set_clim(vmin=0, vmax=1)
  axes[0].set_yticks(
      ticks=range(disrnn_config.latent_size),
      labels=latent_names,
      fontsize=small,
  )
  axes[0].set_xticks(ticks=[])
  axes[0].set_ylabel('Latent #', fontsize=medium)
  axes[0].set_title('Latent Bottlenecks', fontsize=medium)

  # Plot Choice Bottlenecks on axes[1]
  # These bottlenecks apply to the inputs of the choice network:
  # [subject embeddings, latents]
  choice_input_dim = subject_embedding_size + disrnn_config.latent_size
  choice_input_names = np.concatenate((
      [f'SubjEmb {i+1}' for i in range(subject_embedding_size)],
      [f'Latent {i}' for i in latent_names]
  ))
  im2 = axes[1].imshow(np.swapaxes([1 - choice_sigmas], 0, 1), cmap='Oranges')
  im2.set_clim(vmin=0, vmax=1)
  axes[1].set_yticks(
      ticks=range(choice_input_dim), labels=choice_input_names, fontsize=small
  )
  axes[1].set_xticks(ticks=[])
  axes[1].set_ylabel('Choice Network Input', fontsize=medium)
  axes[1].set_title('Choice Network Bottlenecks', fontsize=medium)

  # Plot Confidence Bottlenecks on axes[2]
  # These bottlenecks apply to the inputs of the confidence network:
  # [subject embeddings, latents]
  #confidence_input_dim = subject_embedding_size + disrnn_config.latent_size
  #confidence_input_names = np.concatenate((
  #    [f'SubjEmb {i+1}' for i in range(subject_embedding_size)],
  #    [f'Latent {i}' for i in latent_names]
  #))
  im3 = axes[2].imshow(np.swapaxes([1 - confidence_sigmas], 0, 1), cmap='Oranges')
  im3.set_clim(vmin=0, vmax=1)
  axes[2].set_yticks(
      ticks=range(choice_input_dim), labels=choice_input_names, fontsize=small
  )
  axes[2].set_xticks(ticks=[])
  axes[2].set_ylabel('Confidence Network Input', fontsize=medium)
  axes[2].set_title('Confidence Network Bottlenecks', fontsize=medium)

  # Plot Update Bottlenecks on axes[3]
  im4 = axes[3].imshow(1 - update_sigmas, cmap='Oranges')
  im4.set_clim(vmin=0, vmax=1)
  cbar = fig.colorbar(im3, ax=axes[3])
  # Y-axis corresponds to the target latent (sorted if sort_latents=True)
  cbar.ax.tick_params(labelsize=small)
  axes[3].set_yticks(
      ticks=range(disrnn_config.latent_size),
      labels=latent_names,
      fontsize=small,
  )
  # X-axis corresponds to the inputs to the update network:
  # [subject embeddings, observations, latents]
  xlabels = update_input_names + [f'Latent {i}' for i in latent_names]
  axes[3].set_xticks(
      ticks=range(len(xlabels)),
      labels=xlabels,
      rotation='vertical',
      fontsize=small,
  )
  axes[3].set_ylabel('Latent #', fontsize=medium)
  axes[3].set_xlabel('Update Network Inputs', fontsize=medium)
  axes[3].set_title('Update Network Bottlenecks', fontsize=medium)
  fig.tight_layout()  # Adjust layout to prevent overlap
  return fig

def plot_confidence_rule(
    params: hk.Params,
    disrnn_config: DisRnnWTwoOutputConfig,
    subj_embedding: Optional[np.ndarray] = None,
    axis_lim: float = 2.1,
) -> Optional[plt.Figure]:
  """Plots the confidence rule of a DisRNN.

  Args:
    params: The parameters of the DisRNN
    disrnn_config: A DisRnnWTwoOutputConfig object
    subj_embedding: The subject embedding to use. If None, use a zero vector
      (loosely: the average subject)
    axis_lim: The axis limit for the plot.

  Returns:
    A matplotlib Figure object, or None if choice depends on no latents.
  """

  disrnn_config = copy.deepcopy(disrnn_config)
  disrnn_config.noiseless_mode = True  # Turn off noise for plotting
  activation_fn = getattr(jax.nn, disrnn_config.activation)

  if isinstance(disrnn_config, multisubject_disrnn.MultisubjectDisRnnConfig):
    subj_embedding_size = disrnn_config.subject_embedding_size
    params_prefix = 'multisubject_dis_rnn'
    choice_subj_s = disrnn.reparameterize_sigma(
        params[params_prefix]['choice_net_subj_sigma_params']
    )
    choice_latent_s = disrnn.reparameterize_sigma(
        params[params_prefix]['choice_net_latent_sigma_params']
    )
    choice_net_sigmas = np.concatenate((choice_subj_s, choice_latent_s))
  elif isinstance(disrnn_config, DisRnnWTwoOutputConfig):
    subj_embedding_size = 0
    params_prefix = 'hk_two_output_disentangled_rnn'
    confidence_net_sigmas = disrnn.reparameterize_sigma(
        params[params_prefix]['confidence_net_sigma_params']
    )
  else:
    raise ValueError(
        'DisRnnConfig is neither MultisubjectDisRnnConfig nor DisRnnWTwoOutputsConfig,'
        f' but got {type(disrnn_config).__name__}'
    )

  if subj_embedding is None:
    subj_embedding = np.zeros(shape=(subj_embedding_size,))

  n_vals = 100

  def forward(xs):
    confidence_net_output = disrnn.ResMLP(
        input_size=disrnn_config.latent_size + subj_embedding_size,
        output_size=disrnn_config.output_size,
        n_units_per_layer=disrnn_config.confidence_net_n_units_per_layer,
        n_layers=disrnn_config.confidence_net_n_layers,
        activation_fn=activation_fn,
        name='confidence_net',
    )(xs)
    return confidence_net_output

  model = hk.transform(forward)
  apply = jax.jit(model.apply)

  confidence_net_params = {
      'confidence_net': params[params_prefix + '/confidence_net']
  }

  # Determine which latents to vary based on their confidence_net_sigma_params.
  # confidence_net_sigmas has shape (subj_embedding_size + latent_size,).
  latent_to_confidence_net_sigmas = confidence_net_sigmas[subj_embedding_size:]

  sorted_latent_indices = np.argsort(
      latent_to_confidence_net_sigmas
  )

  influential_latents_indices_in_latent_space = [
      latent_idx  # This is an index in the latent space (0 to latent_size-1)
      for latent_idx in sorted_latent_indices
      if latent_to_confidence_net_sigmas[latent_idx] < 0.5
  ]
  n_latents_to_plot = min(len(influential_latents_indices_in_latent_space), 2)

  if n_latents_to_plot == 0:
    print(
        'Choice rule: No latents have a choice_net_input_sigma < 0.5.'
        ' Plotting not possible.'
    )
    return None

  # Select the actual latents to vary (indices within the latent space)
  varying_latents_plot_indices = influential_latents_indices_in_latent_space[
      :n_latents_to_plot
  ]

  if n_latents_to_plot == 1:
    # Choice Rule 1D: A curve
    policy_latent_idx_in_latent_space = varying_latents_plot_indices[0]
    policy_latent_vals = np.linspace(-axis_lim, axis_lim, n_vals)
    xs = np.zeros((
        n_vals,
        subj_embedding_size + disrnn_config.latent_size,
    ))
    xs[:, :subj_embedding_size] = subj_embedding
    # Vary the selected latent; other latents remain 0
    xs[
        :, subj_embedding_size + policy_latent_idx_in_latent_space
    ] = policy_latent_vals
    confidence_net_output = apply(confidence_net_params, jax.random.PRNGKey(0), xs)
    y_hats = confidence_net_output[0]
    confidence_logits = y_hats[:, 1] - y_hats[:, 0]

    fig, ax = plt.subplots()
    ax.plot(policy_latent_vals, confidence_logits, 'g')
    ax.set_title('Confidence Rule', fontsize=large)
    ax.set_xlabel(
        f'Latent {policy_latent_idx_in_latent_space + 1}', fontsize=medium
    )
    ax.set_ylabel('Confidence Logit', fontsize=medium)
    ax.tick_params(axis='both', labelsize=small)

  else:
    # Confidence Rule 2D: A colormap
    if len(influential_latents_indices_in_latent_space) > 2:
      print(
          'WARNING: More than two latents have choice_net_input_sigma < 0.5.'
          ' Plotting only the two with the smallest choice_net_input_sigmas.'
      )

    policy_latent_idx1_in_latent_space = varying_latents_plot_indices[0]
    policy_latent_idx2_in_latent_space = varying_latents_plot_indices[1]

    latent_vals = np.linspace(-axis_lim, axis_lim, n_vals)

    xv, yv = np.meshgrid(latent_vals, latent_vals)
    latent0_vals = xv.flatten()
    latent1_vals = yv.flatten()

    xs = np.zeros((
        n_vals * n_vals,
        subj_embedding_size + disrnn_config.latent_size,
    ))
    xs[:, :subj_embedding_size] = subj_embedding
    # Vary the selected latents; other latents remain 0
    xs[
        :, subj_embedding_size + policy_latent_idx1_in_latent_space
    ] = latent0_vals
    xs[
        :, subj_embedding_size + policy_latent_idx2_in_latent_space
    ] = latent1_vals

    y_hats = apply(confidence_net_params, jax.random.PRNGKey(0), xs)
    # TODO(kevinjmiller): This assumes two-alternative logits. Generalize to
    # K-alternative choice. For now, take difference between first two.
    confidence_logits_2d = y_hats[0][:, 1] - y_hats[0][:, 0]
    confidence_logits_2d = confidence_logits_2d.reshape((n_vals, n_vals))

    fig, ax = plt.subplots()
    scatter = ax.imshow(
        confidence_logits_2d,
        cmap='coolwarm',
        origin='lower',
        extent=[-axis_lim, axis_lim, -axis_lim, axis_lim],
        aspect='auto',
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.ax.tick_params(labelsize=small)
    cbar.set_label('Confidence Logit', fontsize=medium)
    ax.set_title('Confidence Rule', fontsize=large)
    ax.set_xlabel(
        f'Latent {policy_latent_idx1_in_latent_space + 1}', fontsize=medium
    )
    ax.set_ylabel(
        f'Latent {policy_latent_idx2_in_latent_space + 1}', fontsize=medium
    )
    ax.tick_params(axis='both', labelsize=small)

  return fig


def plot_choice_rule(
    params: hk.Params,
    disrnn_config: DisRnnWTwoOutputConfig,
    axis_lim: float = 2.1,
) -> Optional[plt.Figure]:
  """Plots the choice rule of a DisRNN with confidence prediction."""

  params = {
      key.replace('hk_two_output_disentangled_rnn', 'hk_disentangled_rnn'): value
      for key, value in params.items()
  }
  plotting.plot_choice_rule(
      params=params, disrnn_config=disrnn_config, axis_lim=axis_lim
  )


def plot_update_rules(
    params: hk.Params,
    disrnn_config: DisRnnWTwoOutputConfig,
    axis_lim: float = 2.1,
) -> Optional[plt.Figure]:
  """Plots the update rules of a DisRNN with neural_activity prediction."""
  params = {
      key.replace('hk_two_output_disentangled_rnn', 'hk_disentangled_rnn'): value
      for key, value in params.items()
  }
  plotting.plot_update_rules(
      params=params, disrnn_config=disrnn_config, axis_lim=axis_lim
  )


def log_bottlenecks(
    params: hk.Params,
    open_thresh: float = 0.1,
    partially_open_thresh: float = 0.25,
    closed_thresh: float = 0.9,
) -> dict[str, int]:
  """Computes info about bottlenecks."""
  bnecks = disrnn.log_bottlenecks(
      params, open_thresh, partially_open_thresh, closed_thresh
  )

  params_disrnn = params['hk_disentangled_rnn']

  two_output_sigmas = np.array(
      disrnn.reparameterize_sigma(
          np.transpose(params_disrnn['confidence_net_sigma_params'])
      )
  )

  two_output_bottlenecks_open = np.sum(
      two_output_sigmas < open_thresh
  )
  two_output_bottlenecks_partial = np.sum(
      two_output_sigmas < partially_open_thresh
  )
  two_output_bottlenecks_closed = np.sum(
      two_output_sigmas > closed_thresh
  )

  bottleneck_dict = {
      **bnecks,
      'two_output_bottlenecks_open': int(two_output_bottlenecks_open),
      'two_output_bottlenecks_partial': int(
          two_output_bottlenecks_partial
      ),
      'two_output_bottlenecks_closed': int(
          two_output_bottlenecks_closed
      ),
  }
  return bottleneck_dict


def get_total_sigma(params):
  """Get sum of reparameterized sigmas of a DisRNN."""

  prev_sigma_total = disrnn.get_total_sigma(params)

  params_disrnn = params['hk_disentangled_rnn']

  neural_activity_bottlenecks = disrnn.reparameterize_sigma(
      params_disrnn['confidence_sigma_params']
  )

  return float(jnp.sum(two_output_bottlenecks) + prev_sigma_total)


def get_auxiliary_metrics(
    params: hk.Params,
    make_model_fn: Callable[[], hk.RNNCore],
    dataset_train: rnn_utils.DatasetRNN,
    dataset_eval: rnn_utils.DatasetRNN,
) -> dict[str, np.ndarray]:
  """Compute auxiliary metrics for DisRNN with confidence."""
  xs, ys = dataset_train.get_all()
  network_outputs, _ = rnn_utils.eval_network(
      make_model_fn,
      params,
      xs,
  )
  y_hats = network_outputs[:, :, :-1]
  choice_normlik_train = rnn_utils.normalized_likelihood_and_mse(
      ys, y_hats, likelihood_weight=1.0
  )
  neural_activity_mse_train = rnn_utils.normalized_likelihood_and_mse(
      ys, y_hats, likelihood_weight=0.0
  )

  xs, ys = dataset_eval.get_all()
  network_outputs, _ = rnn_utils.eval_network(
      make_model_fn,
      params,
      xs,
  )
  y_hats = network_outputs[:, :, :-1]
  choice_normlik_eval = rnn_utils.normalized_likelihood_and_mse(
      ys, y_hats, likelihood_weight=1.0
  )
  confidence_mse_eval = rnn_utils.normalized_likelihood_and_mse(
      ys, y_hats, likelihood_weight=0.0
  )

  bottleneck_metrics = log_bottlenecks(params)
  total_sigma_val = get_total_sigma(params)
  total_sigma_val = np.array(total_sigma_val)

  two_output_specific_metrics = {
      'choice_normlik_train': np.array(choice_normlik_train),
      'choice_normlik_eval': np.array(choice_normlik_eval),
      'confidence_mse_train': np.array(confidence_mse_train),
      'confidence_mse_eval': np.array(confidence_mse_eval),
  }
  return {
      **two_output_specific_metrics,
      **bottleneck_metrics,
      'total_sigma': total_sigma_val,
  }
