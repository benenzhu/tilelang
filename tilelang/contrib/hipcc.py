# pylint: disable=invalid-name
"""Utility to invoke hipcc compiler in the system"""
# File is copied from a modified version of hipcc.py to support
# compilation of HIP code with hipcc compiler
# Source Path:
# https://github1s.com/TileLang/tvm/blob/upstream/python/tvm/contrib/hipcc.py

from __future__ import absolute_import as _abs

import hashlib
import logging
import os
import subprocess

import tvm_ffi

from tvm.contrib import utils
from tvm.base import py_str
from tvm.contrib.rocm import get_rocm_arch, find_rocm_path

from tilelang import env as _tl_env

logger = logging.getLogger(__name__)


def compile_hip(code, target_format="hsaco", arch=None, options=None, path_target=None, verbose=False):
    """Compile HIP code with hipcc.

    Parameters
    ----------
    code : str
        The HIP code.

    target_format : str
        The target format of hipcc compiler.

    arch : str
        The AMD GPU architecture.

    options : str or list of str
        The additional options.

    path_target : str, optional
        Output file.

    Return
    ------
    hsaco : bytearray
        The bytearray of the hsaco
    """
    if arch is None:
        rocm_path = find_rocm_path()
        arch = get_rocm_arch(rocm_path)

    if target_format not in ["hsaco"]:
        raise ValueError("target_format must be hsaco")

    save_temps = _tl_env.should_save_compile_temps()
    if save_temps:
        # Persist intermediates next to the HSACO so users can inspect
        # them. Hash by source so repeated compiles of the same code
        # share a directory instead of accumulating tmpXXXX dirs.
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
        tmp_dir = os.path.join(_tl_env.TILELANG_CACHE_DIR, "hipcc_tmp", code_hash)
        os.makedirs(tmp_dir, exist_ok=True)
    else:
        # Match the original tvm.contrib.utils.tempdir() lifetime so the
        # dir is removed when the temp object goes out of scope.
        temp = utils.tempdir()
        tmp_dir = temp.path

    temp_code = os.path.join(tmp_dir, "my_kernel.cc")
    temp_target = os.path.join(tmp_dir, f"my_kernel.{target_format}")

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    file_target = path_target if path_target else temp_target
    cmd = ["hipcc"]
    cmd += ["-O3", "-c"]
    if isinstance(arch, str):
        cmd += [f"--offload-arch={arch}"]
    if target_format == "hsaco":
        cmd += ["--genco"]
    if save_temps:
        # See LibraryGenerator.compile_lib for the same flags.
        cmd += ["--save-temps", "-g"]
    if options:
        if isinstance(options, str):
            cmd += [options]
        elif isinstance(options, list):
            cmd += options
        else:
            raise ValueError("options must be str or list of str")

    cmd += ["-o", file_target]
    cmd += [temp_code]

    # When --save-temps is on, pin TMPDIR + cwd so hipcc/clang's
    # mkstemp-style intermediates land in tmp_dir instead of /tmp + the
    # caller's CWD.
    popen_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
    if save_temps:
        sub_env = dict(os.environ)
        sub_env["TMPDIR"] = tmp_dir
        popen_kwargs.update(cwd=tmp_dir, env=sub_env)
    proc = subprocess.Popen(cmd, **popen_kwargs)

    (out, _) = proc.communicate()
    if verbose:
        print(py_str(out))

    if save_temps:
        try:
            with open(os.path.join(tmp_dir, "hipcc_compile.log"), "wb") as log_f:
                log_f.write(out)
        except OSError:
            logger.exception("Failed to write hipcc save-temps log")

    if proc.returncode != 0:
        msg = code
        msg += "\nCompilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data


@tvm_ffi.register_global_func("tilelang_callback_hip_compile", override=True)
def tilelang_callback_hip_compile(code, target):
    """use hipcc to generate fatbin code for better optimization"""
    from tilelang.utils.target import target_get_mcpu

    hsaco = compile_hip(code, target_format="hsaco", arch=target_get_mcpu(target))
    return hsaco
