from __future__ import annotations
from tvm import tir, IRModule
from tvm.target import Target
import tilelang
from tilelang.transform import PassContext
from tilelang.contrib.nvcc import have_tma, is_hopper, have_pdl

import os


def _is_hip_target(target: Target) -> bool:
    return target.kind.name == "hip"


def _should_interleave_g2s(pass_ctx: PassContext | None = None) -> bool:
    """Check if G2S/MFMA interleaving is enabled."""
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_INTERLEAVE_G2S, False))


def should_print_ir_when_change(pass_ctx: PassContext | None = None) -> bool:
    """Check if IR should be printed when a pass causes changes."""
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_PRINT_IR_WHEN_CHANGE, False))

_print_pass_cnt = 0
_print_pass_mod_origin = None


def print_pass(mod: IRModule, pass_name: str) -> None:
    """
    Record the IR before/after a pass for debugging.
    Should be called right after applying a pass.

    Parameters:
        mod: The module after the pass has been applied
        pass_name: Name of the pass that was just applied
    """
    global _print_pass_cnt, _print_pass_mod_origin

    print_when_change = should_print_ir_when_change()
    print_when_change = True
    if not print_when_change:
        return

    if _print_pass_mod_origin is None:
        # First call: we don't have a "before" snapshot yet, just record current state
        _print_pass_mod_origin = mod.script(show_meta=False)
        return

    if _print_pass_cnt == 0:
        os.system("rm -rf before after")
        os.system("mkdir -p before after")

    filename = f"pass_{_print_pass_cnt:02d}_{pass_name}.py".replace("(", "-").replace(")", "-")
    with open("before/" + filename, "w") as f:
        f.write(_print_pass_mod_origin)
    _print_pass_mod_origin = mod.script(show_meta=False)
    with open("after/" + filename, "w") as f:
        f.write(_print_pass_mod_origin)

    _print_pass_cnt += 1


def allow_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    # avoid circular import
    from tilelang.jit.adapter.utils import is_cuda_target

    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if (not is_cuda_target(target)) or (not have_tma(target)):
        return False
    disable_warp_specialized = pass_ctx.config.get("tl.disable_warp_specialized", False)
    return not disable_warp_specialized


def allow_tma_and_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if not have_tma(target):
        return False
    disable_tma_lower = pass_ctx.config.get("tl.disable_tma_lower", False)
    return not disable_tma_lower and allow_warp_specialized(pass_ctx=pass_ctx, target=target)


def allow_fence_proxy(target: Target | None = None) -> bool:
    return have_tma(target)


def allow_vectorize(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    disable_vectorize = pass_ctx.config.get("tir.disable_vectorize", False)
    return not disable_vectorize


def allow_global_thread_synchronization(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_global_thread_sync = pass_ctx.config.get("tir.detect_global_barrier", False)
    return enable_global_thread_sync


def should_enable_aggressive_merge(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_aggressive_merge = bool(pass_ctx.config.get(tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE, False))
    if allow_warp_specialized(pass_ctx=pass_ctx, target=target):
        # This is a workaround to avoid the bug in the MergeSharedMemoryAllocations pass
        # when warp specialization is enabled, as different warp threads may access different
        # buffers, but the liveness analysis is hard because we need to do pipeline.
        enable_aggressive_merge = False
    return enable_aggressive_merge


def should_force_let_inline(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_FORCE_LET_INLINE, False))


def should_enable_ast_print(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_AST_PRINT_ENABLE, False))


def should_enable_layout_visual(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enabled = pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE, False)
    return enabled


def should_enable_race_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enabled = not pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK, False)
    return enabled


def get_layout_visual_formats(pass_ctx: PassContext | None = None) -> list[str]:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    formats_value = pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS, "")
    if not formats_value:
        return ["txt"]

    formats_str = formats_value.strip().lower()
    valid_formats = ["txt", "png", "pdf", "svg", "all"]

    if formats_str == "all":
        return ["txt", "png", "pdf", "svg"]

    if "," in formats_str:
        formats_list = [f.strip() for f in formats_str.split(",")]
    else:
        formats_list = [formats_str]

    invalid_formats = [f for f in formats_list if f not in valid_formats]
    if invalid_formats:
        raise ValueError(
            f"Invalid formats for TL_LAYOUT_VISUALIZATION_FORMATS: {invalid_formats}. "
            f"Valid formats are: {valid_formats}. "
            f"You can choose one of the valid formats or a comma-separated list of formats.(e.g., 'txt,png,pdf')"
        )
    return formats_list


def LayoutVisual(mod: IRModule) -> None:
    """Apply layout visualization pass if enabled."""
    if should_enable_layout_visual():
        formats = get_layout_visual_formats()
        tilelang.analysis.LayoutVisual(formats=formats)(mod)


def PreLowerSemanticCheck(mod: IRModule) -> None:
    """
    Check whether the module is valid before lowering. If not, raise a user-friendly error
    in Python side instead of letting the error dive into the complicated TVM/C++ stack.
    Note: This is a validation-only pipeline of passes and does not modify or return the module.
    """

    # Print AST for debugging purpose
    if should_enable_ast_print():
        tilelang.analysis.ASTPrinter()(mod)
    # Check if there are any invalid nested loops.
    tilelang.analysis.NestedLoopChecker()(mod)
    # Check if there are any invalid symbolic T.Parallel + fragment access.
    tilelang.analysis.FragmentLoopChecker()(mod)


def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    # Bind the target device information to the module
    """
    Bind target information and progressively legalize and lower frontend Tile IR into a form suitable for downstream optimization and codegen.

    This pass pipeline:
    - Binds the provided target to the module.
    - Legalizes frontend Tile IR into TVM-compatible constructs.
    - Simplifies expressions.
    - Configures reducer layouts and performs layout inference for fragments and shared memory.
    - Lowers high-level tile operations and L2 persistent maps.
    - Legalizes vectorized loops and inserts safety checks for memory accesses.
    - Re-simplifies to remove redundancies introduced by safety checks.
    - Attempts loop vectorization for dynamic-shaped loops.

    Parameters:
        mod (IRModule): The input IR module containing frontend Tile IR.
        target (Target): Target device information to bind into the module.

    Returns:
        IRModule: The transformed module, ready for target-specific optimization passes.
    """
    mod = tir.transform.BindTarget(target)(mod)
    print_pass(mod, "BindTarget")

    if should_force_let_inline():
        # Force-let inline whenever the pass config requests it.
        mod = tilelang.transform.LetInline()(mod)
        print_pass(mod, "LetInline")
    # Add wrapper for single buf store
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    print_pass(mod, "AddWrapperForSingleBufStore")
    # Normalize negative indices to canonical non-negative form
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    # Verify parallel loop correctness
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)

    # Inject assumes to speedup tvm prover
    mod = tilelang.transform.InjectAssumes()(mod)
    print_pass(mod, "InjectAssumes")
    # Simplify the IR expressions
    mod = tilelang.transform.Simplify()(mod)
    print_pass(mod, "Simplify")
    # Set layouts for reducers
    mod = tilelang.transform.LayoutReducer()(mod)
    print_pass(mod, "LayoutReducer")
    # Infer memory layouts for fragments and shared memory
    mod = tilelang.transform.LayoutInference()(mod)
    print_pass(mod, "LayoutInference")
    # Visualize the layout
    LayoutVisual(mod)
    # Lower high-level tile operations to low-level operations
    mod = tilelang.transform.LowerTileOp()(mod)
    print_pass(mod, "LowerTileOp")
    # Lower l2 persistent map
    mod = tilelang.transform.LowerL2Persistent()(mod)
    # Decouple type cast vectorization constraints before vectorization
    mod = tilelang.transform.DecoupleTypeCast()(mod)
    # Legalize vectorized loops to ensure they are valid
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    print_pass(mod, "LegalizeVectorizedLoop")
    # Add safety checks for memory accesses
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    print_pass(mod, "LegalizeSafeMemoryAccess")
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    # use an enhanced pass to simplify the dynamic symbolics
    # TODO(lei): return to tir pass when kSymbolicBound simplification
    # is merged into tvm.
    mod = tilelang.transform.Simplify()(mod)
    print_pass(mod, "Simplify")
    # Hoist any root-block annotations to PrimFunc attrs if pass is available
    mod = tilelang.transform.HoistNonRestrictParams()(mod)
    print_pass(mod, "HoistNonRestrictParams")
    return mod


def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    pass_ctx = tilelang.transform.get_pass_context()
    # Lower the shared.barrier into specific initialization slot
    mod = tilelang.transform.LowerSharedBarrier()(mod)
    print_pass(mod, "LowerSharedBarrier")
    # Lower the shared.tmem into specific initialization slot
    mod = tilelang.transform.LowerSharedTmem()(mod)
    print_pass(mod, "LowerSharedTmem")
    # which may be introduced by the LegalizeSafeMemoryAccess
    if allow_tma_and_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.IfStmtBinding()(mod)
        print_pass(mod, "IfStmtBinding")
        mod = tilelang.transform.MultiVersionBuffer()(mod)
        print_pass(mod, "MultiVersionBuffer")
        mod = tilelang.transform.WarpSpecialized()(mod)
        print_pass(mod, "WarpSpecialized")
        mod = tilelang.transform.InjectTmaBarrier()(mod)
        print_pass(mod, "InjectTmaBarrier")
        # if tma is not enabled, we can also do pipeline planning
        # to get better performance with async copy
        mod = tilelang.transform.PipelinePlanning()(mod)
        print_pass(mod, "PipelinePlanning")
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
        print_pass(mod, "InjectSoftwarePipeline")
        # warp_specialized pass will pack the if stmt into the block
        # so we need to lower the opaque block first
        mod = tilelang.transform.LowerOpaqueBlock()(mod)
        print_pass(mod, "LowerOpaqueBlock")
        if is_hopper(target):
            mod = tilelang.transform.RewriteWgmmaSync()(mod)
            print_pass(mod, "RewriteWgmmaSync")
    else:
        mod = tilelang.transform.IfStmtBinding()(mod)
        print_pass(mod, "IfStmtBinding")
        mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        print_pass(mod, "PlanAndUpdateBufferAllocationLocation")
        mod = tilelang.transform.PipelinePlanning()(mod)
        print_pass(mod, "PipelinePlanning")
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
        print_pass(mod, "InjectSoftwarePipeline")

    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    print_pass(mod, "LowerOpaqueBlock")
    mod = tilelang.transform.Simplify()(mod)
    print_pass(mod, "Simplify")

    # Interleave G2S with MFMA compute (HipKittens-style)
    if _is_hip_target(target) and _should_interleave_g2s(pass_ctx):
        mod = tilelang.transform.InterleaveG2SWithCompute()(mod)
        print_pass(mod, "InterleaveG2SWithCompute")

    mod = tir.transform.NarrowDataType(32)(mod)
    print_pass(mod, "NarrowDataType")
    mod = tilelang.transform.FlattenBuffer()(mod)
    print_pass(mod, "FlattenBuffer")
    # ConfigIndexBitwidth must be applied after FlattenBuffer
    # as it will flatten index computing
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    print_pass(mod, "ConfigIndexBitwidth")
    mod = tir.transform.Simplify()(mod)
    print_pass(mod, "tir.Simplify")
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    print_pass(mod, "VectorizeLoop")
    mod = tilelang.transform.StorageRewrite()(mod)
    print_pass(mod, "StorageRewrite")
    mod = tilelang.transform.LoopUnswitching()(mod)
    print_pass(mod, "LoopUnswitching")
    mod = tilelang.transform.UnrollLoop()(mod)
    print_pass(mod, "UnrollLoop")
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    print_pass(mod, "RenormalizeSplitPattern")
    mod = tir.transform.Simplify()(mod)
    print_pass(mod, "tir.Simplify")
    mod = tir.transform.RemoveNoOp()(mod)
    print_pass(mod, "RemoveNoOp")
    mod = tir.transform.HoistIfThenElse()(mod)
    print_pass(mod, "HoistIfThenElse")

    mod = tir.transform.VerifyMemory()(mod)
    print_pass(mod, "VerifyMemory")
    mod = tir.transform.AnnotateEntryFunc()(mod)
    print_pass(mod, "AnnotateEntryFunc")
    # TODO(lei): This is a hack to make sure the
    # thread level allreduce pass can be applied
    # in TL. As Tl only use one thread dimension
    # the var binding information will be lost
    # in the lowering process with Legalization
    # and Simplify pass.
    # We can find a way better to create var instead
    # of putting the LowerThreadAllreduce before
    # the Legalization.
    mod = tir.transform.InferFragment()(mod)
    print_pass(mod, "InferFragment")
    mod = tilelang.transform.LowerThreadAllreduce()(mod)
    print_pass(mod, "LowerThreadAllreduce")
    mod = tilelang.transform.LowerLDGSTG()(mod)
    print_pass(mod, "LowerLDGSTG")
    mod = tilelang.transform.LowerHopperIntrin()(mod)
    print_pass(mod, "LowerHopperIntrin")
    # Global Barrier Synchronization must be applied before
    # SplitHostDevice pass, as the global barrier
    if allow_global_thread_synchronization():
        mod = tilelang.transform.ThreadSync("global")(mod)
        print_pass(mod, "ThreadSync(global)")
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    print_pass(mod, "AnnotateDeviceRegions")
    mod = tilelang.transform.SplitHostDevice()(mod)
    print_pass(mod, "SplitHostDevice")

    # Mark the function contains pdl_sync or pdl_trigger
    mod = tilelang.transform.MarkCudaSyncCalls(have_pdl(target))(mod)
    print_pass(mod, "MarkCudaSyncCalls")

    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    print_pass(mod, "AnnotateReadOnlyParams")
    # MergeSharedMemoryAllocations must be applied after SplitHostDevice
    # because the merged allocation site is at the beginning of each device function
    #
    # HIP: skip merging so each shared buffer stays as a separate __shared__
    # variable.  This is required for the truly-async G2S path
    # (buffer_load_b128…lds): LLVM's alias analysis must see the LDS-read
    # target as an "identified object" distinct from the buffer_load_lds
    # intrinsic's dummy AS3-null write pointer.  A single merged buffer
    # (even if declared static) is not enough — only genuinely separate
    # __shared__ allocations achieve NoAlias, preventing the compiler from
    # inserting spurious s_waitcnt vmcnt(0) before ds_read instructions.
    if not _is_hip_target(target):
        enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
        mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge)(mod)
        print_pass(mod, "MergeSharedMemoryAllocations")
    if allow_tma_and_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.InjectFenceProxy()(mod)
        print_pass(mod, "InjectFenceProxy")
    else:
        if allow_fence_proxy(target=target):
            # in hopper device, wgmma is an async proxy
            # so we need to inject a fence proxy before it
            mod = tilelang.transform.InjectFenceProxy()(mod)
            print_pass(mod, "InjectFenceProxy")
    mod = tilelang.transform.ThreadSync("shared")(mod)
    print_pass(mod, "ThreadSync(shared)")
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    print_pass(mod, "ThreadSync(shared.dyn)")
    mod = tilelang.transform.MergeIfStmt()(mod)
    print_pass(mod, "MergeIfStmt")
    # Inject PTX async copy must behind the thread sync pass
    # as ptx async copy won't be recognized as a valid buffer load
    mod = tilelang.transform.InjectPTXAsyncCopy()(mod)
    print_pass(mod, "InjectPTXAsyncCopy")
    if allow_tma_and_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.AnnotateWarpGroupRegAlloc()(mod)
        print_pass(mod, "AnnotateWarpGroupRegAlloc")
    mod = tilelang.transform.MakePackedAPI()(mod)
    print_pass(mod, "MakePackedAPI")
    mod = tilelang.transform.Simplify()(mod)
    print_pass(mod, "Simplify")
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    print_pass(mod, "LowerDeviceKernelLaunch")

    # Transform threadblock to persistent threadblock
    mod = tilelang.transform.PersistThreadblock()(mod)
    print_pass(mod, "PersistThreadblock")

    return mod
