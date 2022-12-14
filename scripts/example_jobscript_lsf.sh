#!/usr/bin/bash

#BSUB -n 1
#BSUB -gpu "num=1"
#BSUB -R "span[ptile=32]"
#BSUB -e "output/%J.err"
#BSUB -o "output/%J.out"

export NVCC_WRAPPER_DEFAULT_COMPILER=mpicxx
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

#export KOKKOS_PROFILE_LIBRARY=/home/ntan1/kokkos-tools/profiling/simple-kernel-timer/kp_kernel_timer.so
#export KOKKOS_PROFILE_LIBRARY=/home/ntan1/kokkos-tools/profiling/space-time-stack/kp_space_time_stack.so
#export KOKKOS_PROFILE_LIBRARY=/home/ntan1/kokkos-tools/profiling/memory-events/kp_memory_events.so

#chunk_sizes=(128 256 512 1024 2048 4096)
chunk_sizes=(512)
project_dir=/data/gclab/fido/scratch/checkpoints_HiPC
msg_race_prefix=message_race_nprocs_32_niters_16384_msg_size_32_run_000_size_11174336.edgelist
unst_mesh_prefix=unstructured_mesh_nprocs_32_packed_nd_fraction_0.5_niters_2048_msg_size_32_run_000_nodes_14418368.edgelist
asia_osm_prefix=asia_osm.mtx
num_iter=1
num_threads=16
dedup_approaches=('--run-full-chkpt' '--run-naive-chkpt' '--run-list-chkpt' '--run-tree-chkpt')
#dedup_approaches=('--run-full-chkpt' '--run-tree-chkpt')

make test

for chunk_size in "${chunk_sizes[@]}";
do

  echo "======================================================"
  echo "Deduplicate MD5 Message Race"
  echo "======================================================"
  rm $project_dir/message_race_layout_left/*chkpt
  for approach in "${dedup_approaches[@]}";
  do
    ./dedup_chkpt_files $chunk_size 10 $approach \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-0.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-1.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-2.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-3.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-4.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-5.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-6.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-7.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-8.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-9.dat \
      --kokkos-num-threads=2 --kokkos-map-device-id-by=random
    sleep 30s
  done


  echo "======================================================"
  echo "Restart MD5 Message Race"
  echo "======================================================"
  for chkpt in {0..9..9}
  do
    for approach in "${dedup_approaches[@]}";
    do
    ./restart_chkpt_files $chkpt 10 $num_iter $chunk_size $approach \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-0.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-1.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-2.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-3.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-4.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-5.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-6.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-7.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-8.dat \
      $project_dir/message_race_layout_left/$msg_race_prefix-0-9.dat \
      --kokkos-num-threads=4 --kokkos-map-device-id-by=random
      sleep 30s
    done
  done

  echo "======================================================"
  echo "Deduplicate MD5 Unstructured Mesh"
  echo "======================================================"
  rm $project_dir/unstructured_mesh_layout_left/*chkpt
  for approach in "${dedup_approaches[@]}";
  do
    ./dedup_chkpt_files $chunk_size 10 $approach \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-0.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-1.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-2.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-3.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-4.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-5.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-6.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-7.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-8.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-9.dat \
      --kokkos-num-threads=2 --kokkos-map-device-id-by=random
    sleep 30s
  done

  echo "======================================================"
  echo "Restart MD5 Unstructured Mesh"
  echo "======================================================"
  for chkpt in {0..9..9}
  do
    for approach in "${dedup_approaches[@]}";
    do
    ./restart_chkpt_files $chkpt 10 $num_iter $chunk_size $approach \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-0.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-1.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-2.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-3.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-4.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-5.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-6.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-7.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-8.dat \
      $project_dir/unstructured_mesh_layout_left/$unst_mesh_prefix-0-9.dat \
      --kokkos-num-threads=4 --kokkos-map-device-id-by=random 
      sleep 30s
    done
  done

  echo "======================================================"
  echo "Deduplicate MD5 Asia OSM"
  echo "======================================================"
  rm $project_dir/asia_osm_layout_left/*chkpt
  for approach in "${dedup_approaches[@]}";
  do
    ./dedup_chkpt_files $chunk_size 10 $approach \
      $project_dir/asia_osm_layout_left/$asia_osm_prefix-0-0.dat \
      $project_dir/asia_osm_layout_left/$asia_osm_prefix-0-1.dat \
      $project_dir/asia_osm_layout_left/$asia_osm_prefix-0-2.dat \
      $project_dir/asia_osm_layout_left/$asia_osm_prefix-0-3.dat \
      $project_dir/asia_osm_layout_left/$asia_osm_prefix-0-4.dat \
      $project_dir/asia_osm_layout_left/$asia_osm_prefix-0-5.dat \
      $project_dir/asia_osm_layout_left/$asia_osm_prefix-0-6.dat \
      $project_dir/asia_osm_layout_left/$asia_osm_prefix-0-7.dat \
      $project_dir/asia_osm_layout_left/$asia_osm_prefix-0-8.dat \
      $project_dir/asia_osm_layout_left/$asia_osm_prefix-0-9.dat \
      --kokkos-num-threads=2 --kokkos-map-device-id-by=random
    sleep 30s
  done


  echo "======================================================"
  echo "Restart MD5 Asia OSM"
  echo "======================================================"
  for chkpt in {0..9..9}
  do
    for approach in "${dedup_approaches[@]}";
    do
      ./restart_chkpt_files $chkpt 10 $num_iter $chunk_size $approach \
        $project_dir/asia_osm_layout_left/asia_osm.mtx-0-0.dat \
        $project_dir/asia_osm_layout_left/asia_osm.mtx-0-1.dat \
        $project_dir/asia_osm_layout_left/asia_osm.mtx-0-2.dat \
        $project_dir/asia_osm_layout_left/asia_osm.mtx-0-3.dat \
        $project_dir/asia_osm_layout_left/asia_osm.mtx-0-4.dat \
        $project_dir/asia_osm_layout_left/asia_osm.mtx-0-5.dat \
        $project_dir/asia_osm_layout_left/asia_osm.mtx-0-6.dat \
        $project_dir/asia_osm_layout_left/asia_osm.mtx-0-7.dat \
        $project_dir/asia_osm_layout_left/asia_osm.mtx-0-8.dat \
        $project_dir/asia_osm_layout_left/asia_osm.mtx-0-9.dat \
        --kokkos-num-threads=4 --kokkos-map-device-id-by=random
      sleep 30s
    done
  done
done

