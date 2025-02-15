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


This project demonstrates parallel image processing using MPI and OpenCV, focusing on distributing image data among multiple processors to apply filters and transformations in parallel. The lab consists of four parts:

Serial implementation of a convolution kernel.

Parallel distribution and collection of numerical data with halos using MPI.

Image segmentation, modification, and reassembly using MPI.

Parallel image blurring using convolution and MPI.

Setup Instructions

Ensure you have MPI installed (e.g., OpenMPI or MPICH).

Install OpenCV for image processing.

Place pic.jpg in the compilation folder before running the program.

Compile the provided C++ files using an MPI-enabled compiler.

Project Structure

Part 1: Serial Convolution Kernel

Converts a given convolution kernel into a serial implementation.

Removes OpenMP parallelization for compatibility.

Uses OpenCV functions to display images before and after applying the filter.

Run with: ./lab2_part1 <filename> <blur_level>

Part 2: Distributed Numerical Data Processing with Halos

Distributes a dataset across multiple MPI processes.

Each process receives a section of data along with halo elements for boundary computations.

Utilizes MPI_Scatterv for distribution and MPI_Gatherv for collection.

Usage: Run the executable with multiple processors.

Part 3: Image Segmentation and Reconstruction

Splits an image into multiple sections with halo rows.

Each process modifies its assigned portion of the image.

Uses MPI_Scatterv and MPI_Gatherv for efficient data transfer.

Outputs individual and merged image results.

Usage: mpirun -np <num_processes> ./lab2_part3

Part 4: Parallel Image Blurring

Divides an image into sections for parallel convolution.

Applies a blur kernel to each section.

Merges the modified sections while preserving image integrity.

Uses OpenCV to compare results with a serial implementation.

Usage: mpirun -np <num_processes> ./lab2_part4

Performance Analysis

Speedup Comparison

Performance was measured by running the blur and sharpen filters on a 2560x1440 image using 1-8 processors. Key findings:

Parallelization improves execution time, but speedup is not always linear due to overhead.

Larger images benefit more from parallel processing.

Guided scheduling generally provides better load balancing compared to static or dynamic scheduling.

Kernel Effects

Two different convolution kernels were tested:

Laplacian Kernel (Sharpening) - Enhances edges by approximating the second spatial derivative.

Gaussian Blur Kernel - Smoothens the image by averaging neighboring pixel values.

Alternative Partitioning Strategies

Besides row-based partitioning, the following approaches could be considered:

Column Partitioning - Useful for images with non-uniform height-to-width ratios.

Block Partitioning - Divides the image into small square regions, ensuring better parallel load distribution.

Tile Partitioning - Uses small fixed-size tiles assigned dynamically to processes for better scalability.

Execution Instructions

To compile and run any part of the project:

mpic++ -o lab2_partX lab2_partX.cpp `pkg-config --cflags --libs opencv4`
mpirun -np <num_processes> ./lab2_partX

Replace X with the part number (1-4) and adjust <num_processes> accordingly.

Conclusion

This lab provided hands-on experience with MPI's scatterv and gatherv functions for distributing and collecting image data. By leveraging parallelism, image processing tasks such as blurring and segmentation were significantly optimized. The experiments confirmed that effective partitioning and scheduling strategies are critical for maximizing performance in parallel image processing.
