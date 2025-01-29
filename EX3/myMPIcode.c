#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  // Initialize MPI
  MPI_Init(&argc, &argv);
  
  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  // Get the processor name (hostname)
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  
  // Print a greeting from each process
  printf("Hello from process %d out of %d processes on host %s!\n",
         world_rank, world_size, processor_name);
  
  // Start the timer
  double start_time = MPI_Wtime();
  
  // Define a large array for demonstration (on rank 0)
  const int N = 100000; // Array size
  int *array = NULL;
  if (world_rank == 0) {
    array = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
      array[i] = i + 1; // Fill the array with numbers 1 to N
    }
  }
  
  // Divide the array among processes
  int chunk_size = N / world_size;
  int *sub_array = (int*)malloc(chunk_size * sizeof(int));
  
  // Scatter the array to all processes
  MPI_Scatter(array, chunk_size, MPI_INT, sub_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
  
  // Calculate the local sum of the sub-array
  int local_sum = 0;
  for (int i = 0; i < chunk_size; i++) {
    local_sum += sub_array[i];
  }
  
  // Reduce all local sums to a global sum on process 0
  int global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  
  // End the timer
  double end_time = MPI_Wtime();
  
  // Process 0 prints the result and execution time
  if (world_rank == 0) {
    printf("Global sum = %d\n", global_sum);
    printf("Execution time = %f seconds\n", end_time - start_time);
    free(array); // Free the array on process 0
  }
  
  // Clean up
  free(sub_array);
  MPI_Finalize();
  
  return 0;
}