# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""

import sys
import gc

import torch
from torch.nn.parallel import DistributedDataParallel as torchDDP

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from megatron import (
    get_args,
    get_adlr_autoresume,
)
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.model.module import param_is_not_shared

import pandas as pd
from mup import get_shapes, make_base_shapes
from mup.coord_check import plot_coord_data, _record_coords

def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                if args.bf16:
                    params_data.append(param.data.float())
                else:
                    params_data.append(param.data)
    # Calculate norm
    dummy_overflow_buf = torch.cuda.IntTensor([0])
    norm, _ = multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        dummy_overflow_buf,
        [params_data],
        False # no per-parameter norm
    )
    norm_2 = norm * norm
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(norm_2,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=mpu.get_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)

def save_base_shapes(base_shapes_filename):

    from megatron.core.enums import ModelType
    from megatron.model import GPTModel
    from megatron.arguments import core_transformer_config_from_args
    from megatron.training import get_model

    # take ModelType from pretrain_gpt.py
    model_type = ModelType.encoder_or_decoder

    # get base model shape
    # base_model is instantiated as in pretrain_gpt.py
    base_args = get_args()
    base_config = core_transformer_config_from_args(base_args)
    base_model = lambda pre_process=True, post_process=True: GPTModel(
        base_config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    base_model = get_model(base_model, model_type)
    assert len(base_model) == 1, 'For saving muP shapes, base_model is expected to be a list of length 1'
    base_shapes = get_shapes(base_model[0])
    del base_model

    # get delta model shapes corresponding to the base model scaled by `delta_scaling_factor` 
    # across width dimensions: hidden_size, ffn_hidden_size, kv_channels
    delta_args = get_args()
    delta_scaling_factor = 2

    # scale width dimensions
    # setting `ffn_hidden_size` and `kv_channels` to None will trigger their derivation in __post_init__() of `TransformerConfig` 
    # NB: make sure that latter is consistent with the duplicated definition in `arguments.py`
    delta_args.hidden_size *= delta_scaling_factor
    delta_args.ffn_hidden_size = None 
    delta_args.kv_channels = None 

    delta_config = core_transformer_config_from_args(delta_args)
    delta_model = lambda pre_process=True, post_process=True: GPTModel(
        delta_config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    delta_model = get_model(delta_model, model_type)
    assert len(delta_model) == 1, 'For saving muP shapes, delta_model should be a list of length 1'
    delta_shapes = get_shapes(delta_model[0])
    del delta_model

    # derive and save infshapes and its dimensionalitites from base_shapes and delta_shapes
    make_base_shapes(base_shapes, delta_shapes, savefile=f'{base_shapes_filename}')
    print('Saved base shapes to', base_shapes_filename)


def _get_coord_data(setups, data_iterator, nsteps=3, nseeds=1, 
                    filter_module_by_name=None,
                    output_fdict=None, input_fdict=None, param_fdict=None):
    '''
    Inner method for `get_coord_data`. This is the adaptation of the corresponding function
        from `mup.coord_check` to the `Megatron-LM` setup.

    Train the models in `models` with optimizer given by `optcls` and data from
    `dataloader` for `nsteps` steps, and record coordinate statistics specified
    by `output_fdict`, `input_fdict`, `param_fdict`. By default, only `l1` is
    computed for output activations of each module.
    
    '''
    
    from megatron.core.enums import ModelType
    from megatron.training import setup_model_and_optimizer, train_step
    from pretrain_gpt import forward_step

    df = []
    for i in range(nseeds):
        torch.manual_seed(i)

        for width, (model_provider, model_config) in setups.items():
            
            # Instantiate model and optimiser
            model_type = ModelType.encoder_or_decoder
            model, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, model_type, use_mup=model_config.use_mup)
            assert len(model) == 1, 'For _get_coord_data(), model should be a list of length 1'

            # Turn on training mode which enables dropout.
            for model_module in model:
                model_module.train()

            step = 0
            consumed_train_samples = 0
            while step < nsteps:
                # add hooks to record per-layer statistics as defined by `_record_coords()`
                remove_hooks = []
                for name, module in model[0].named_modules():
                    if filter_module_by_name and not filter_module_by_name(name):
                        continue
                    remove_hooks.append(
                        module.register_forward_hook(
                            _record_coords(df, width, name, step+1,
                                output_fdict=output_fdict,
                                input_fdict=input_fdict,
                                param_fdict=param_fdict
                            )
                        )
                    )
                    
                # train for a step
                loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                    train_step(forward_step,
                            data_iterator,
                            model,
                            optimizer,
                            opt_param_scheduler,
                            model_config)
                step += 1

                for handle in remove_hooks:
                    handle.remove()

            del model
            gc.collect()

    return pd.DataFrame(df)


def mup_coord_check(width_dimensions, data_iterator, nsteps, nseeds, save_dir, legend=False):

    def setup_provider(hidden_size, use_mup):

        # override base args with `hidden_size` and `use_mup`
        # setting `ffn_hidden_size` and `kv_channels` to None will derive them in __post_init__() of `TransformerConfig`
        _args = get_args()
        _args.hidden_size = hidden_size
        _args.ffn_hidden_size = None
        _args.kv_channels = None
        _args.use_mup = use_mup

        # return lazy model with updated args
        model_config = core_transformer_config_from_args(_args)
        lazy_model = lambda pre_process=True, post_process=True: GPTModel(
            model_config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )

        return (lazy_model, model_config)

    from megatron.model import GPTModel
    from megatron.arguments import core_transformer_config_from_args
    
    args = get_args()

    # create dictionary with lazy models+configs for each width and muP/SP
    setups_mup, setups_sp = {}, {}
    for width in width_dimensions:
        setups_mup[width] = setup_provider(width, use_mup=True)
        setups_sp[width] = setup_provider(width, use_mup=False)

    # collect statistics into dataframes for each model width & seed across `nsteps` 
    df_mup = _get_coord_data(setups_mup, data_iterator, nsteps, nseeds)
    df_mup.to_csv(f'{save_dir}/coord_check_muP.{torch.distributed.get_rank()}.csv')
    
    # plot check results and save
    plot_coord_data(df_mup, legend=legend,
                    save_to=f"{save_dir}/coord_check_muP.{torch.distributed.get_rank()}.jpg",
                    suptitle=f'muP Transformer {args.optimizer} lr={args.lr} nseeds={nseeds}',
                    face_color='xkcd:light grey'
                    ) 
