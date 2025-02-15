ECE 4530 - Lab 2: Embarrassingly Parallel Image Processing

Origional Image:
![image](https://github.com/user-attachments/assets/d61e8dfb-2d08-431e-b453-b7a56559d7b0)
Distributed Image:

![image](https://github.com/user-attachments/assets/c4fdc4d3-d1c4-44cc-8907-28eb4c546283)

Merged Image:
![image](https://github.com/user-attachments/assets/48642448-5390-42d6-b45b-b5f31d1aa0fc)


Distributed Blurred Image:
![image](https://github.com/user-attachments/assets/4aa0a457-5e8a-4a58-8902-3a92a035f6f8)

Merged Blurred Image:
![image](https://github.com/user-attachments/assets/17330cbe-4adb-417f-9447-49b8f7ee4318)

Overview
Overview

This project explores parallel image processing techniques using MPI and OpenCV. The lab focuses on distributing and manipulating image data across multiple processors for performance optimization. Key techniques include:

    Scatterv & Gatherv for data distribution and collection.
    Halo exchange for handling image borders.
    Parallel convolution kernels for image processing.

Requirements

    MPI (Message Passing Interface)
    OpenCV (for image processing)
    C++ Compiler supporting MPI (e.g., mpic++)
    Test Image (pic.jpg) placed in the compilation directory

Lab Components
1. Serial Convolution Kernel (Lab2_1.cpp)

    Converts an OpenMP-based convolution kernel into a serial algorithm.
    Uses OpenCV to load and display an image before and after blurring.
    Takes image filename and blur level as input.
    Example usage:

    ./Lab2_1 pic.jpg 5

    Outputs the processed image with the specified blur level.

2. Data Distribution with Halos (Lab2_2.cpp)

    Demonstrates scatterv/gatherv for distributing and recollecting data while handling halo exchange.
    Works with arbitrary halo sizes and processor counts (≥2).
    Uses MPI data types to ensure efficient communication.

Example Outputs:

    With 8 processors, halo = 3
    With 3 processors, halo = 5

3. Parallel Image Segmentation (Lab2_3.cpp)

    Splits an image into segments distributed among processors.
    Each segment is color-coded according to processor rank.
    Uses MPI_Bcast, scatterv, and gatherv for communication.
    Displays the processed image and reconstructs it at rank 0.

4. Parallel Image Blurring (Lab2_4.cpp)

    Distributes an image, applies a blur, and then merges the results.
    Implements a convolution kernel similar to Part 1.
    MPI-based parallelization ensures each processor applies the filter on its segment.
    The halo region is NOT blurred, ensuring seamless merging.

Example Outputs:

    Distributed Blurred Image
    Merged Blurred Image
    Comparison to Serial Blurring (serial_blurred.jpg)

Execution Instructions
Compiling the Code

To compile the programs:

mpic++ -o Lab2_1 Lab2_1.cpp `pkg-config --cflags --libs opencv4`
mpic++ -o Lab2_2 Lab2_2.cpp
mpic++ -o Lab2_3 Lab2_3.cpp `pkg-config --cflags --libs opencv4`
mpic++ -o Lab2_4 Lab2_4.cpp `pkg-config --cflags --libs opencv4`

Running the Programs

Run using mpirun with the desired number of processors:

mpirun -np 4 ./Lab2_2
mpirun -np 3 ./Lab2_3
mpirun -np 4 ./Lab2_4 pic.jpg

Performance Analysis
Speedup Analysis

Speedup was measured using 1 to 8 cores for sharpening and blurring kernels.
Processors	Sharpening Time (s)	Speedup	Blurring Time (s)	Speedup
1	0.735	1.00	4.82	1.00
2	0.671	1.09	4.42	1.09
4	0.515	1.42	4.26	1.13
8	0.581	1.26	3.12	1.54
Observations

    Speedup increases with core count but is not linear due to overhead.
    Blurring takes longer than sharpening due to the larger convolution operation.
    Larger images benefit more from parallelization.

Alternative Partitioning Strategies

Instead of static row partitioning, other approaches could be:

    Column Partitioning: Each processor handles vertical slices.
        Pros: Similar to row partitioning.
        Cons: No major advantage unless aspect ratio favors vertical slicing.

    Block Partitioning: Divides the image into a grid of blocks.
        Pros: Better balance across processors.
        Cons: Requires more complex data exchange.

    Tile-Based Partitioning: Assigns small tiles dynamically to processors.
        Pros: Handles regions of varying complexity well.
        Cons: Higher scheduling overhead.
