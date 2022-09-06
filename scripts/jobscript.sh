#!/usr/bin/bash

#BSUB -n 1
#BSUB -gpu "num=1"
#BSUB -e "output/%J.err"
#BSUB -o "output/%J.out"

#datalen=1073741824
#datalen=2147483648
#chunk_sizes=(512 1024 2048 4096)
#gen_chunk_sizes=(1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912)
chunk_sizes=( 128 ) # 256 512 1024)
#chunk_sizes=(64 128 256 512 1024 2048 4096 8192 12288 16384 20480 24576 28672 32768 36864 40960 45056 49152 53248 57344 61440 65536 69632 73728 77824 81920)
#gen_chunk_sizes=(134217728 268435456 402653184 536870912 671088640 805306368 939524096)
#num_chkpts=8

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
#export KOKKOS_PROFILE_LIBRARY=$HOME/kokkos-tools/profiling/simple-kernel-timer/kp_kernel_timer.so
#export KOKKOS_PROFILE_LIBRARY=$HOME/kokkos-tools/profiling/space-time-stack/kp_space_time_stack.so
#for i in {0..0}
#do
#  ./merkle_tree_test --kokkos-threads=4 --kokkos-num-devices=1
#  ./merkle_tree_test_files 1 testfile0.txt testfile1.txt --kokkos-threads=4 --kokkos-num-devices=1
#done

#./merkle_tree_test_files 2 2 \
#  testfile0.txt \
#  testfile1.txt \
#  --kokkos-num-threads=32
#
for chunk_size in "${chunk_sizes[@]}";
do
#for i in {1..5} 
#do

#./merkle_tree_test_files $chunk_size 10 \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-0.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-1.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-2.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-3.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-4.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-5.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-6.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-7.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-8.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-9.dat \
#  --kokkos-num-threads=4 --kokkos-num-devices=1
#./merkle_tree_test_files $chunk_size 10 \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-0.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-1.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-2.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-3.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-4.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-5.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-6.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-7.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-8.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-9.dat \
#  --kokkos-num-threads=4 --kokkos-num-devices=1
#./merkle_tree_test_files $chunk_size 10 \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-0.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-1.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-2.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-3.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-4.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-5.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-6.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-7.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-8.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-9.dat \
#  --kokkos-num-threads=4 --kokkos-num-devices=1

#./merkle_tree_test_files 1 3 \
#  testfile0.txt \
#  testfile1.txt \
#  testfile2.txt \
#  --kokkos-num-threads=4 --kokkos-num-devices=1

for chkpt in {0..9}
do
./test_restart $chkpt 10 \
  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-0.dat \
  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-1.dat \
  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-2.dat \
  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-3.dat \
  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-4.dat \
  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-5.dat \
  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-6.dat \
  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-7.dat \
  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-8.dat \
  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-9.dat \
  --kokkos-num-threads=4 --kokkos-num-devices=1
done
#for chkpt in {0..9}
#do
#./test_restart $chkpt 10 \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-0.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-1.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-2.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-3.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-4.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-5.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-6.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-7.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-8.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-9.dat \
#  --kokkos-num-threads=4 --kokkos-num-devices=1
#done
#for chkpt in {0..9}
#do
#./test_restart $chkpt 10 \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-0.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-1.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-2.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-3.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-4.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-5.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-6.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-7.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-8.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/asia_osm_layout_left/asia_osm.mtx-0-9.dat \
#  --kokkos-num-threads=4 --kokkos-num-devices=1
#done
#./test_restart 0 3 \
#  testfile0.txt \
#  testfile1.txt \
#  testfile2.txt \
#  --kokkos-num-threads=4 --kokkos-num-devices=1
#./test_restart 1 3 \
#  testfile0.txt \
#  testfile1.txt \
#  testfile2.txt \
#  --kokkos-num-threads=4 --kokkos-num-devices=1
#./test_restart 2 3 \
#  testfile0.txt \
#  testfile1.txt \
#  testfile2.txt \
#  --kokkos-num-threads=4 --kokkos-num-devices=1
#done
done
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-0.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-1.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-2.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-3.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-4.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-5.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-6.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-7.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-8.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-9.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-0.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-1.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-2.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-3.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-4.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-5.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-6.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-7.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-8.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/unstructured_mesh_layout_left/unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist-0-9.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-0.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-1.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-2.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-3.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-4.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-5.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-6.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-7.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-8.dat \
#  /data/gclab/fido/scratch/checkpoints_HiPC/message_race_layout_left/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-9.dat \

#for chunk_size in "${chunk_sizes[@]}";
#do
#for i in {1..5} 
#do
#  ./merkle_tree_test_files $chunk_size \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-0.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-1.dat \
#    --kokkos-threads=4 --kokkos-num-devices=1

#  ./merkle_tree_test_files $chunk_size \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-0.dat \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-1.dat \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-2.dat \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-3.dat \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-4.dat \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-5.dat \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-6.dat \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-7.dat \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-8.dat \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-9.dat \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-10.dat \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-11.dat \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-12.dat \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-13.dat \
#    /data/gclab/fido/scratch/message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist-0-14.dat \
#    --kokkos-num-threads=4

#  ./merkle_tree_test_files $chunk_size \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-0.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-1.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-2.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-3.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-4.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-5.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-6.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-7.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-8.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-9.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-10.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-11.dat \
#    /data/gclab/fido/scratch/ecology_layout_left_trim_chkpts/ecology1.mtx-0-12.dat \
#    --kokkos-threads=4 --kokkos-num-devices=1

#  ./merkle_tree_test_files $chunk_size \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-0.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-1.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-2.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-3.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-4.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-5.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-6.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-7.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-8.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-9.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-10.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-11.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-12.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-13.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-14.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-15.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-16.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-17.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-18.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-19.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-20.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-21.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-22.dat \
#    /data/gclab/fido/scratch/asia_osm_layout_left_trim_chkpts/asia_osm.mtx-0-23.dat \
#    --kokkos-num-threads=8

#  ./merkle_tree_test_files $chunk_size \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-0.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-1.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-2.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-3.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-4.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-5.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-6.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-7.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-8.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-9.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-10.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-11.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-12.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-13.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-14.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-15.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-16.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-17.dat \
#    /data/gclab/fido/scratch/message_race_layout_left_trim_chkpts/message_race_nprocs_32_niters_2048_msg_size_32_run_000_size_1397184.edgelist-0-18.dat \
#    --kokkos-threads=4 --kokkos-num-devices=1

#  ./merkle_tree_test_files $chunk_size \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-0.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-1.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-2.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-3.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-4.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-5.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-6.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-7.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-8.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-9.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-10.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-11.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-12.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-13.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_layout_left_trim_chkpts/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-14.dat \
#    --kokkos-threads=4 --kokkos-num-devices=1

#  ./merkle_tree_test_files $chunk_size \
#    /data/gclab/fido/scratch/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-0.dat \
#    /data/gclab/fido/scratch/unstructured_mesh_nprocs_32_packed_nd_fraction_0.25_niters_1024_msg_size_32_run_000_nodes_7209408.edgelist-0-1.dat \
#    --kokkos-threads=4 --kokkos-num-devices=1


#  mv thetagpu*.dat chunk_size_${chunk_size}.dat
#  rename thetagpu "chunk_size_${chunk_size}" thetagpu*.dat
#done
#done

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

##gen_chunk_sizes=(1024 16384 262144 4194304 67108864)
#gen_chunk_sizes=(1024)
#echo "======================================================"
#echo "Sparse"
#echo "======================================================"
#for chunk_size in "${chunk_sizes[@]}";
#do
#  for n_changes in "${gen_chunk_sizes[@]}";
#  do
#    echo "******************************************************"
#    echo "Chunk size: ${chunk_size}, Num changes: ${n_changes}"
#    echo "******************************************************"
#    for i in {0..4}
#    do
#      echo "Run ${i}" >> hashtree_len_${datalen}_chunk_size_${chunk_size}_n_chkpts_${num_chkpts}_generator_mode_S_chance_${n_changes}.csv
#      echo "Run ${i}" >> hashlist_len_${datalen}_chunk_size_${chunk_size}_n_chkpts_${num_chkpts}_generator_mode_S_chance_${n_changes}.csv
#      ./merkle_tree_test $chunk_size $datalen $n_changes $num_chkpts S --kokkos-threads=4 --kokkos-num-devices=1
#      echo "" >> hashtree_len_${datalen}_chunk_size_${chunk_size}_n_chkpts_${num_chkpts}_generator_mode_S_chance_${n_changes}.csv
#      echo "" >> hashlist_len_${datalen}_chunk_size_${chunk_size}_n_chkpts_${num_chkpts}_generator_mode_S_chance_${n_changes}.csv
##      mv tellico-compute*.dat sparse_datalen_${datalen}_chunksize_${chunk_size}_nchkpts_${num_chkpts}_mode_I_chance_${n_changes}_sample_${i}.dat
#    done
#  done
#done

