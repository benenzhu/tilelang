from __future__ import annotations
from tvm import tir, IRModule
from tvm.target import Target
import tvm
import tilelang
from tilelang.transform import PassContext
from tilelang.contrib.nvcc import have_tma, is_hopper, have_pdl
import os


def should_print_ir_when_change(pass_ctx: PassContext | None = None) -> bool:
    """Check if IR should be printed when a pass causes changes."""
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_PRINT_IR_WHEN_CHANGE, False))

cnt = 0
mod_origin = None

def apply_pass(mod: IRModule, transform_pass, pass_name: str | None = None) -> IRModule:
    """
    Apply a pass to the module and optionally print IR if it changed.
    
    Parameters:
        mod: The input IR module
        transform_pass: The pass to apply (can be a pass object or a callable that returns a pass)
        pass_name: Optional name for the pass (auto-detected if not provided)
    
    Returns:
        The transformed module
    """
    global mod_origin
    if mod_origin is None:
        mod_origin = mod.script(show_meta=False)
    print_when_change = should_print_ir_when_change()
    print_when_change=True
    assert print_when_change==True, "print_when_change must be True"
    if not print_when_change:
        # Fast path: just apply the pass
        return transform_pass(mod)
    
    # Get pass name
    if pass_name is None:
        if hasattr(transform_pass, 'info'):
            pass_name = transform_pass.info().name
        elif hasattr(transform_pass, '__name__'):
            pass_name = transform_pass.__name__
        else:
            pass_name = str(type(transform_pass).__name__)
    
    # Compute hash before
    before_hash = tvm.ir.structural_hash(mod)
    
    # Apply the pass
    new_mod = transform_pass(mod)
    
    # Compute hash after
    after_hash = tvm.ir.structural_hash(new_mod)
    
    # Print if changed
    global cnt

    # if before_hash != after_hash:
    #     print(f"\n{'='*70}")
    #     print(f"[have change]=== IR CHANGED by: {pass_name} ===")
    #     print(f"{'='*70}")
    #     # print(new_mod.script())
    # else:
    #     print(f"[No change] {pass_name}")
    if cnt == 0: 
        os.system("rm -rf before after")
        os.system('mkdir -p before after')
    filename = f"pass_{cnt:02d}_{pass_name}.py".replace("(", "-").replace(")", "-")
    with open("before/" + filename, "w") as f: 
        f.write(mod_origin)
    mod_origin = new_mod.script(show_meta=False)
    with open("after/" + filename, "w") as f:
        f.write(mod_origin)
    
    cnt += 1
    
    return new_mod


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
    mod = apply_pass(mod, tir.transform.BindTarget(target), "BindTarget")

    if should_force_let_inline():
        # Force-let inline whenever the pass config requests it.
        mod = apply_pass(mod, tilelang.transform.LetInline(), "LetInline")
    # Add wrapper for single buf store
    mod = apply_pass(mod, tilelang.transform.AddWrapperForSingleBufStore(), "AddWrapperForSingleBufStore")
    # Normalize negative indices to canonical non-negative form
    mod = apply_pass(mod, tilelang.transform.LegalizeNegativeIndex(), "LegalizeNegativeIndex")
    # Inject assumes to speedup tvm prover
    mod = apply_pass(mod, tilelang.transform.InjectAssumes(), "InjectAssumes")
    # Simplify the IR expressions
    mod = apply_pass(mod, tilelang.transform.Simplify(), "Simplify")
    # Set layouts for reducers
    mod = apply_pass(mod, tilelang.transform.LayoutReducer(), "LayoutReducer")
    # Infer memory layouts for fragments and shared memory
    mod = apply_pass(mod, tilelang.transform.LayoutInference(), "LayoutInference")
    # Visualize the layout
    LayoutVisual(mod)
    # Lower high-level tile operations to low-level operations
    mod = apply_pass(mod, tilelang.transform.LowerTileOp(), "LowerTileOp")
    # Lower l2 persistent map
    mod = apply_pass(mod, tilelang.transform.LowerL2Persistent(), "LowerL2Persistent")
    # Legalize vectorized loops to ensure they are valid
    mod = apply_pass(mod, tilelang.transform.LegalizeVectorizedLoop(), "LegalizeVectorizedLoop")
    # Add safety checks for memory accesses
    mod = apply_pass(mod, tilelang.transform.LegalizeSafeMemoryAccess(), "LegalizeSafeMemoryAccess")
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    # use an enhanced pass to simplify the dynamic symbolics
    # TODO(lei): return to tir pass when kSymbolicBound simplification
    # is merged into tvm.
    mod = apply_pass(mod, tilelang.transform.Simplify(), "Simplify")
    # Hoist any root-block annotations to PrimFunc attrs if pass is available
    mod = apply_pass(mod, tilelang.transform.HoistNonRestrictParams(), "HoistNonRestrictParams")
    return mod


def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    pass_ctx = tilelang.transform.get_pass_context()
    # Lower the barrier.arrive into specific initialization slot
    mod = apply_pass(mod, tilelang.transform.LowerSharedBarrier(), "LowerSharedBarrier")
    # Lower the shared.tmem into specific initialization slot
    mod = apply_pass(mod, tilelang.transform.LowerSharedTmem(), "LowerSharedTmem")
    # which may be introduced by the LegalizeSafeMemoryAccess
    if allow_tma_and_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = apply_pass(mod, tilelang.transform.IfStmtBinding(), "IfStmtBinding")
        mod = apply_pass(mod, tilelang.transform.MultiVersionBuffer(), "MultiVersionBuffer")
        mod = apply_pass(mod, tilelang.transform.WarpSpecialized(), "WarpSpecialized")
        mod = apply_pass(mod, tilelang.transform.InjectTmaBarrier(), "InjectTmaBarrier")
        # if tma is not enabled, we can also do pipeline planning
        # to get better performance with async copy
        mod = apply_pass(mod, tilelang.transform.PipelinePlanning(), "PipelinePlanning")
        mod = apply_pass(mod, tilelang.transform.InjectSoftwarePipeline(), "InjectSoftwarePipeline")
        # warp_specialized pass will pack the if stmt into the block
        # so we need to lower the opaque block first
        mod = apply_pass(mod, tilelang.transform.LowerOpaqueBlock(), "LowerOpaqueBlock")
        mod = apply_pass(mod, tilelang.transform.MergeIfStmt(), "MergeIfStmt")
        if is_hopper(target):
            mod = apply_pass(mod, tilelang.transform.RewriteWgmmaSync(), "RewriteWgmmaSync")
        mod = apply_pass(mod, tilelang.transform.InjectFenceProxy(), "InjectFenceProxy")
    else:
        mod = apply_pass(mod, tilelang.transform.IfStmtBinding(), "IfStmtBinding")
        mod = apply_pass(mod, tilelang.transform.PlanAndUpdateBufferAllocationLocation(), "PlanAndUpdateBufferAllocationLocation")
        mod = apply_pass(mod, tilelang.transform.PipelinePlanning(), "PipelinePlanning")
        mod = apply_pass(mod, tilelang.transform.InjectSoftwarePipeline(), "InjectSoftwarePipeline")
        mod = apply_pass(mod, tilelang.transform.MergeIfStmt(), "MergeIfStmt")
        if allow_fence_proxy(target=target):
            # in hopper device, wgmma is an async proxy
            # so we need to inject a fence proxy before it
            mod = apply_pass(mod, tilelang.transform.InjectFenceProxy(), "InjectFenceProxy")

    mod = apply_pass(mod, tilelang.transform.LowerOpaqueBlock(), "LowerOpaqueBlock")
    mod = apply_pass(mod, tilelang.transform.Simplify(), "Simplify")
    mod = apply_pass(mod, tir.transform.NarrowDataType(32), "NarrowDataType")
    mod = apply_pass(mod, tilelang.transform.FlattenBuffer(), "FlattenBuffer")
    # ConfigIndexBitwidth must be applied after FlattenBuffer
    # as it will flatten index computing
    mod = apply_pass(mod, tilelang.transform.ConfigIndexBitwidth(), "ConfigIndexBitwidth")
    mod = apply_pass(mod, tir.transform.Simplify(), "tir.Simplify")
    mod = apply_pass(mod, tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx)), "VectorizeLoop")
    mod = apply_pass(mod, tilelang.transform.StorageRewrite(), "StorageRewrite")
    mod = apply_pass(mod, tir.transform.UnrollLoop(), "UnrollLoop")
    mod = apply_pass(mod, tir.transform.RenormalizeSplitPattern(), "RenormalizeSplitPattern")
    mod = apply_pass(mod, tir.transform.Simplify(), "tir.Simplify")
    mod = apply_pass(mod, tir.transform.RemoveNoOp(), "RemoveNoOp")
    mod = apply_pass(mod, tir.transform.RewriteUnsafeSelect(), "RewriteUnsafeSelect")
    mod = apply_pass(mod, tir.transform.HoistIfThenElse(), "HoistIfThenElse")

    mod = apply_pass(mod, tir.transform.VerifyMemory(), "VerifyMemory")
    mod = apply_pass(mod, tir.transform.AnnotateEntryFunc(), "AnnotateEntryFunc")
    # TODO(lei): This is a hack to make sure the
    # thread level allreduce pass can be applied
    # in TL. As Tl only use one thread dimension
    # the var binding information will be lost
    # in the lowering process with Legalization
    # and Simplify pass.
    # We can find a way better to create var instead
    # of putting the LowerThreadAllreduce before
    # the Legalization.
    mod = apply_pass(mod, tir.transform.InferFragment(), "InferFragment")
    mod = apply_pass(mod, tilelang.transform.LowerThreadAllreduce(), "LowerThreadAllreduce")

    mod = apply_pass(mod, tilelang.transform.LowerHopperIntrin(), "LowerHopperIntrin")
    # Global Barrier Synchronization must be applied before
    # SplitHostDevice pass, as the global barrier
    if allow_global_thread_synchronization():
        mod = apply_pass(mod, tilelang.transform.ThreadSync("global"), "ThreadSync(global)")
    mod = apply_pass(mod, tilelang.transform.AnnotateDeviceRegions(), "AnnotateDeviceRegions")
    mod = apply_pass(mod, tilelang.transform.SplitHostDevice(), "SplitHostDevice")

    # Mark the function contains pdl_sync or pdl_trigger
    mod = apply_pass(mod, tilelang.transform.MarkCudaSyncCalls(have_pdl(target)), "MarkCudaSyncCalls")

    mod = apply_pass(mod, tilelang.transform.AnnotateReadOnlyParams(), "AnnotateReadOnlyParams")
    # MergeSharedMemoryAllocations must be applied after SplitHostDevice
    # because the merged allocation site is at the beginning of each device function
    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    mod = apply_pass(mod, tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge), "MergeSharedMemoryAllocations")
    mod = apply_pass(mod, tilelang.transform.ThreadSync("shared"), "ThreadSync(shared)")
    mod = apply_pass(mod, tilelang.transform.ThreadSync("shared.dyn"), "ThreadSync(shared.dyn)")
    # Inject PTX async copy must behind the thread sync pass
    # as ptx async copy won't be recognized as a valid buffer load
    mod = apply_pass(mod, tilelang.transform.InjectPTXAsyncCopy(), "InjectPTXAsyncCopy")
    if allow_tma_and_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = apply_pass(mod, tilelang.transform.AnnotateWarpGroupRegAlloc(), "AnnotateWarpGroupRegAlloc")
    mod = apply_pass(mod, tilelang.transform.MakePackedAPI(), "MakePackedAPI")
    mod = apply_pass(mod, tilelang.transform.Simplify(), "Simplify")
    mod = apply_pass(mod, tilelang.transform.LowerDeviceKernelLaunch(), "LowerDeviceKernelLaunch")

    # Transform threadblock to persistent threadblock
    mod = apply_pass(mod, tilelang.transform.PersistThreadblock(), "PersistThreadblock")

    return mod
