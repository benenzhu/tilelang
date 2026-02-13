rm -rf outdir
ZZPERF=1 rocprofv3 --att -d outdir -- python example_gemm.py
tar -czvf a.tar.gz outdir/ui*_3