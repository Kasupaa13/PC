from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import timeit


# Visualize the butterfly sum communication pattern
def plot_butterfly_sum():
    fig, ax = plt.subplots()
    steps = int(np.ceil(np.log2(size)) + 1)
    for step in range(steps):
        for proc in range(size):
            ax.scatter(step + 1, proc, color='blue')
            if step > 0:
                partner = proc ^ (1 << (step - 1))
                if partner < size:
                    ax.plot([step, step + 1], [partner, proc], 'k-')
    ax.set_title('Butterfly Sum Communication')
    ax.set_xlabel('Step')
    ax.set_ylabel('Process')
    ax.set_yticks(range(size))
    ax.set_xticks(range(1, steps + 1))
    plt.show()


# Butterfly sum for parallel reduction
"""
This function performs the butterfly sum. The idea is the same as in the slides.
Therefore, we use ^ to perform XOR as stated. Since step is actually 2**(level-1),
we can simply do rank XOR step. 
Sendrecv is a convenience function that does both a send and a receive. In our case,
we know that every process sends and receives each level, so this suits the scenario.
"""
def butterfly_sum(local):
    step = 1
    while step < size:
        partner = rank ^ step
        if partner < size: # If not, there is not partner this level
            recv_value = np.empty(len(local), np.float64) # Create space for receiving
            comm.Sendrecv(sendbuf=local, dest=partner, sendtag=0,
                         recvbuf=recv_value, source=partner, recvtag=0) # A convenient MPI function
            # print(f"On level {int(np.log2(step) + 1)}, rank {rank} sends to rank {partner}.")
            local += recv_value # Add the vector to the one it had already
        step *= 2 # At step i, p communicates with p + 2**(i-1)
    return local


"""
A truly serial implementation of the dot product.
It works for both vector-vector and matrix-vector multiplication.
If we want matrix-vector multiplication, then x1 should be the matrix.
"""
def serial_dot(x1, x2):
    if x1.ndim == 1:
        local_sum = 0
        for x1_el, x2_el in zip(x1, x2):
            local_sum += x1_el*x2_el
        return local_sum
    elif x1.ndim == 2:
        dim = x1.shape[0]
        local_sum = np.empty(dim, np.float64)
        for i in range(dim):
            small_sum = 0
            for x1_el, x2_el in zip(x1[i], x2):
                small_sum += x1_el*x2_el
            local_sum[i] = small_sum
        return local_sum
    else:
        TypeError("First argument not of correct dimensions.")

def parallel_matvec(N, comm, rank, size):
    local_n = N // size  # Number of elements per process

    # Root process initializes full vectors
    if rank == 0:
        x = np.random.rand(N)
        A = np.random.rand(N, N)

        """
        In the following lines, send_A is created. This is the variable that actually
        gets scattered to local_A per process. We do this, because we cannot simply
        scatter a matrix. One option is to transpose the matrix, but we chose to
        create a different format, as this requires no work for the processes that
        are not the root process.
        """
        # Split into sub-arrays along required axis
        arrs = np.split(A, size, axis=1)

        # Flatten the sub-arrays
        raveled = [np.ravel(arr) for arr in arrs]

        # Join them back up into a 1D array
        send_A = np.concatenate(raveled)

        # print(f"Vector x: {x}")
        # print(f"Matrix A: {A}")
        # print(f"Serial result: {serial_dot(A,x)}")

        # For timing the serial matvec
        time_start_serial = timeit.default_timer()
        serial_dot(A,x) # TODO: change between numpy and our own implementation
        time_serial = timeit.default_timer() - time_start_serial

        # Start timer for parallel matvec
        time_begin = timeit.default_timer()
    else:
        x = None
        A = None
        send_A = None
        time_begin = None
        time_serial = None

    # Allocate space for local portions of x and y
    local_x = np.empty(local_n, dtype=np.float64)
    local_A = np.empty((N, local_n), dtype=np.float64)

    # Distribute data using Scatter
    comm.Scatter(x, local_x, root=0)
    comm.Scatter(send_A, local_A, root=0)

    # Compute parallel scalar product
    local_result = serial_dot(local_A, local_x)
    # result = butterfly_sum(local_result) # TODO: butterfly or Allreduce
    result = np.empty(N, np.float64)
    comm.Allreduce(result, local_result, op=MPI.SUM)
    return result, time_begin, time_serial


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define vector size (must be divisible by size for simplicity)
    if rank == 0:
        N = int(input("Enter N:"))  # Total number of elements (we assume it is of the form 2**x)
    else:
        N = None
    N = comm.bcast(N, root=0)
    result, time_begin, time_serial = parallel_matvec(N, comm, rank, size)
    # print(f"Rank {rank} returns {result}")
    if rank == 0:
        # print(f"Final result is {result}")
        # plot_butterfly_sum()
        with open("output.txt", "a") as file:
            file.write(f"{str(size)} {str(time_serial/(timeit.default_timer()-time_begin))}\n")

    # # Strong scaling
    # N_list = [2**x for x in range(int(np.log2(size)), int(np.log2(size) + 10))] # TODO: input
    # if rank == 0:
    #     par_times = []
    #     ser_times = []
    # for N in N_list:
    #     result, time_begin, time_serial = parallel_matvec(N, comm, rank, size)
    #     if rank == 0:
    #         par_times.append(timeit.default_timer() - time_begin)
    #         ser_times.append(time_serial)
    # if rank == 0:
    #     # # for Roos
    #     # for en, speedup in zip(N_list, np.array(ser_times)/np.array(par_times)):
    #     #     print(en, speedup)

    #     plt.plot(N_list, np.array(ser_times)/np.array(par_times))
    #     plt.xlabel("N")
    #     plt.ylabel("Speed-up")
    #     plt.show()

    MPI.Finalize()

    #TODO: How to test weak scaling? Bc we increase N exponentially, and we cannot do that with num of processes.
    #TODO: What expected of sketching algorithm?
    #TODO: I related to drawing. Is that what was meant?