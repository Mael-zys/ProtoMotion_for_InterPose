# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

import typer
import yaml
import random

def main(amass_file: Path, amass_file_additional: Path, out_file: Path, ratio: float=1):
    with open(amass_file_additional, "r") as f:
        additional_motion_file = yaml.load(f, Loader=yaml.SafeLoader)["motions"]
    with open(amass_file, "r") as f:
        amass_motion_file = yaml.load(f, Loader=yaml.SafeLoader)["motions"]

    # merge the two yaml files and fix indices
    final_yaml_dict_format = {"motions": []}
    current_index = 0
    total_duration = 0

    for entry in amass_motion_file:
        for sub_motion in entry["sub_motions"]:
            current_index += 1
            total_duration += sub_motion['timings']['end']
        final_yaml_dict_format["motions"].append(entry)
    

    add_len = len(additional_motion_file)
    if ratio != 1:
        random.shuffle(additional_motion_file)
        add_len = int(add_len * ratio)
    for entry in additional_motion_file[:add_len]:
        entry["idx"] = current_index
        for sub_motion in entry["sub_motions"]:
            sub_motion["weight"] = 1
            sub_motion["idx"] = current_index
            sub_motion['hml3d_id'] = current_index
            current_index += 1
            total_duration += sub_motion['timings']['end']
        final_yaml_dict_format["motions"].append(entry)

    file = open(out_file, "w")
    yaml.dump(final_yaml_dict_format, file)
    file.close()

    print(f"Total motion sequences: {current_index}. Total duration: {total_duration / 3600:.2f} hours")


if __name__ == "__main__":
    typer.run(main)
