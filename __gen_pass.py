from flask import Flask, render_template_string
import re
import os
from collections import OrderedDict
from pathlib import Path
os.system("rm before/* after/*")
cnt = 0
def parse_mlir_file(file_path):
    global cnt
    passes = OrderedDict()
    current_pass = ""
    current_ir = []
    current_state = ""  # 记录是Before还是After

    with open(file_path, 'r', encoding="UTF-8") as f:
        line_num = 0
        __lines = f.readlines()
        while line_num < len(__lines):
            line = __lines[line_num]
            # Match pass header
            pass_match = re.match('--------------------------------------------------------.*', line)
            # pass_match = re.match(r'// -----// IR Dump (Before|After) (.+) //----- //', line)
            if pass_match:
                # Save previous pass if exists
                if current_pass and current_ir:
                    if current_pass not in passes:
                        passes[current_pass] = {}
                    passes[current_pass] = '\n'.join(current_ir)
                    # print(current_ir)
                    # os._exit()
                    current_ir = []
                line_num += 1
                current_state = __lines[line_num].replace("# from tvm.script import ir as I\n", "").replace(" #include <tl_templates/cuda/gemm.h>\n'", "")
                # current_pass = str(cnt) + "_" + pass_match.group(2)
                current_pass = str(cnt) + "_" + Path(current_state).name.replace(' ', '_')
                cnt += 1
            else:
                current_ir.append(line.rstrip())
            line_num += 1

        # Add final pass
        if current_pass and current_ir:
            if current_pass not in passes:
                passes[current_pass] = {}
            passes[current_pass] = '\n'.join(current_ir)

    return passes

passes = parse_mlir_file('pass.py')
passes_names = list(passes.keys())

passes_names = ['_'.join(i.split(" ")) for i in passes]
# for k,v in passes.items(): 
#     print(k, v)
os.system("mkdir -p before after")
cnt = 0
# for i, pass_name in enumerate(passes_names):
#     if passes[pass_name]["Before"]:
#         with open(f"before/{cnt:05d}_before_{pass_name}.mlir", "w") as f:
#             f.write(passes[pass_name]["Before"])
#         cnt += 1

# cnt2 = 0
# for i, pass_name in enumerate(passes_names[1:]):
#     if passes[pass_name]["Before"]:
#         with open(f"after/{cnt2:05d}_before_{passes_names[i]}.mlir", "w") as f:
#             f.write(passes[pass_name]["Before"])
#         cnt2 += 1

# with open(f"after/{cnt2:05d}_before_{passes_names[-1]}.mlir", "w") as f:
#     f.write("")

before = """from tvm.script import ir as I
from tvm.script import tir as T
"""
for i, pass_name in enumerate(passes_names[:-1]): 
    with open(f"before/{i:05d}_{passes_names[i + 1]}.py", "w") as f:
        # print(pass_name, passes[pass_name])
        f.write(before + passes[pass_name])

for i, pass_name in enumerate(passes_names[1:]): 
    with open(f"after/{i:05d}_{pass_name}.py", "w") as f:
        f.write(before + passes[pass_name])