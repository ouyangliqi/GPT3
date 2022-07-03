# BSD 3-Clause License
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the psutil authors nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def get_patrickstar_config(
    args, lr=0.001, betas=(0.9, 0.999), eps=1e-6, weight_decay=0
):
    config = {
        # The same format as optimizer config of DeepSpeed
        # https://www.deepspeed.ai/docs/config-json/#optimizer-parameters
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "use_hybrid_adam": args.use_hybrid_adam,
            },
        },
        "fp16": {
            "enabled": True,
            # Set "loss_scale" to 0 to use DynamicLossScaler.
            "loss_scale": 0,
            "initial_scale_power": args.init_loss_scale_power,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "default_chunk_size": args.default_chunk_size,
        "release_after_init": args.release_after_init,
        "use_fake_dist": args.use_fake_dist,
        "use_cpu_embedding": args.use_cpu_embedding,
        "client": {
            "mem_tracer": {
                "use_async_mem_monitor": args.with_async_mem_monitor,
                "warmup_gpu_chunk_mem_ratio": 0.2,
                "overall_gpu_mem_ratio": 0.9,
                "overall_cpu_mem_ratio": 0.9,
                "margin_use_ratio": 0.8,
                "use_fake_dist": False,
                "with_static_partition": args.with_static_partition,
            },
            "opts": {
                "with_mem_saving_comm": args.with_mem_saving_comm,
                "with_mem_cache": args.with_mem_cache,
                "with_async_move": args.with_async_move,
            },
        },
    }

    return config
