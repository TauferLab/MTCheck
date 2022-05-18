#!/usr/bin/env bash

#BSUB -n 1
#BSUB -R "span[ptile=32]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -e "output/%J.err"
#BSUB -o "output/%J.out"

#datalen=1073741824
datalen=4294957296
#chunk_sizes=(512 1024 2048 4096)
#gen_chunk_sizes=(1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912)
#chunk_sizes=(256 512 1024 2048 4096 8192 16384 32768 65536 131072)
chunk_sizes=(1024 4096 8192 12288 16384 20480 24576 28672 32768 36864 40960 45056 49152 53248 57344 61440 65536 69632 73728 77824 81920)
gen_chunk_sizes=(134217728 268435456 402653184 536870912 671088640 805306368 939524096)
num_chkpts=8

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
#export KOKKOS_PROFILE_LIBRARY=/home/ntan1/kokkos-tools/profiling/simple-kernel-timer/kp_kernel_timer.so
#./merkle_tree_test 1 53 0 2 R --kokkos-threads=4 --kokkos-num-devices=1

#echo "======================================================"
#echo "Random Data"
#echo "======================================================"
#for chunk_size in "${chunk_sizes[@]}";
#do
#    echo "******************************************************"
#    echo "Chunk size: ${chunk_size}"
#    echo "******************************************************"
#    for i in {0..4}
#    do
#      ./merkle_tree_test $chunk_size $datalen 0 $num_chkpts R --kokkos-threads=4 --kokkos-num-devices=1
#      mv tellico-compute*.dat random_datalen_${datalen}_chunksize_${chunk_size}_nchkpts_${num_chkpts}_mode_R_chance_0_sample_${i}.dat
#    done
#done
#
#echo "======================================================"
#echo "Identical Data"
#echo "======================================================"
#for chunk_size in "${chunk_sizes[@]}";
#do
#    echo "******************************************************"
#    echo "Chunk size: ${chunk_size}"
#    echo "******************************************************"
#    for i in {0..4}
#    do
#      ./merkle_tree_test $chunk_size $datalen 0 $num_chkpts I --kokkos-threads=4 --kokkos-num-devices=1
#      mv tellico-compute*.dat identical_datalen_${datalen}_chunksize_${chunk_size}_nchkpts_${num_chkpts}_mode_I_chance_0_sample_${i}.dat
#    done
#done
#
#echo "======================================================"
#echo "Single Chunk"
#echo "======================================================"
#for chunk_size in "${chunk_sizes[@]}";
#do
#  for gen_chunk_len in "${gen_chunk_sizes[@]}";
#  do
#    echo "******************************************************"
#    echo "Chunk size: ${chunk_size}, Generator chunk size: ${gen_chunk_len}"
#    echo "******************************************************"
#    for i in {0..4}
#    do
#      ./merkle_tree_test $chunk_size $datalen $gen_chunk_len $num_chkpts B --kokkos-threads=4 --kokkos-num-devices=1
#      mv tellico-compute*.dat single_chunk_datalen_${datalen}_chunksize_${chunk_size}_nchkpts_${num_chkpts}_mode_I_chance_${gen_chunk_len}_sample_${i}.dat
#    done
#  done
#done

#gen_chunk_sizes=(1024 16384 262144 4194304 67108864)
gen_chunk_sizes=(1024)
echo "======================================================"
echo "Sparse"
echo "======================================================"
for chunk_size in "${chunk_sizes[@]}";
do
  for n_changes in "${gen_chunk_sizes[@]}";
  do
    echo "******************************************************"
    echo "Chunk size: ${chunk_size}, Num changes: ${n_changes}"
    echo "******************************************************"
    for i in {0..5}
    do
      echo "Run ${i}" >> hashtree_len_${datalen}_chunk_size_${chunk_size}_n_chkpts_${num_chkpts}_generator_mode_S_chance_${n_changes}.csv
      echo "Run ${i}" >> hashlist_len_${datalen}_chunk_size_${chunk_size}_n_chkpts_${num_chkpts}_generator_mode_S_chance_${n_changes}.csv
      ./merkle_tree_test $chunk_size $datalen $n_changes $num_chkpts S --kokkos-threads=4 --kokkos-num-devices=1
      echo "" >> hashtree_len_${datalen}_chunk_size_${chunk_size}_n_chkpts_${num_chkpts}_generator_mode_S_chance_${n_changes}.csv
      echo "" >> hashlist_len_${datalen}_chunk_size_${chunk_size}_n_chkpts_${num_chkpts}_generator_mode_S_chance_${n_changes}.csv
#      mv tellico-compute*.dat sparse_datalen_${datalen}_chunksize_${chunk_size}_nchkpts_${num_chkpts}_mode_I_chance_${n_changes}_sample_${i}.dat
    done 
  done
done

