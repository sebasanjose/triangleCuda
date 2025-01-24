# CUDA Triangle Rasterization

This project implements a basic triangle rasterization pipeline using CUDA. It processes vertices and textures to generate a 2D image output. The code optimizes memory usage and parallelism, leveraging constant memory for efficient GPU computations.

## Features

- **Vertex and Triangle Data**: Stored in constant memory for quick, shared access across threads.
- **Texture Sampling**: A custom function maps texture coordinates to texels and extracts normalized RGBA values.
- **Rasterization Pipeline**: Converts 3D vector abstractions into 2D pixels with barycentric interpolation and color blending.
- **Output Image**: Writes the rendered output to a BMP file.

---

## File Overview

### `main.cu`
The main CUDA source file contains the implementation of the rasterization pipeline and BMP image writing.

#### Key Sections:
1. **Vertex and Triangle Data Structures (lines 34–50)**:
   - Stored in constant memory for global accessibility across GPU kernels.
   - Efficient use of memory hierarchy avoids unnecessary transactions.

2. **TextureGPU Structure (lines 46–52)**:
   - Represents texture data in constant memory for easy access.

3. **Light Direction Vector (line 55)**:
   - Stores light direction in GPU-accessible memory.

4. **Triangle Rasterization (lines 123–178)**:
   - Maps 3D vertices to 2D screen space.
   - Uses barycentric coordinates for interpolation and checks if a pixel is inside the triangle.
   - Writes interpolated colors to the output buffer.

5. **BMP Writing Utility (lines 184–239)**:
   - Outputs the rasterized image to a BMP file.
   - Not detailed further in this README, as it is ancillary to the primary focus.

6. **Main Function (lines 268–318)**:
   - Initializes vertex and light data.
   - Copies resources to constant memory.
   - Allocates buffers and launches the rasterization kernel.
   - Saves the final image and cleans up memory.

### `my_helper_math.h`
Utility functions for vector math operations, optimized for use in CUDA. Includes:
- **Operator Overloads**:
  - Vector addition, scalar multiplication for `float2`, `float3`, and `float4`.
- **Math Utilities**:
  - Dot product, normalization, and extracting the `xyz` component of a `float4`.

---

## Memory Hierarchy and Optimization

### **Constant Memory**:
- Vertex, triangle, and texture data are stored here, minimizing global memory transactions.
- Optimized for broadcasting small, read-only data to all threads in a warp.

### **Shared Memory**:
- Not used for vertex/triangle data to avoid redundancy and block-specific limitations.

### **Local Memory**:
- Avoided due to its thread-specific nature and higher overhead.

### **Global Memory**:
- Used for allocating large buffers like the output color array.

---

## How to Run

1. **Build**:
   - Use the NVIDIA CUDA compiler (nvcc) to compile the source files:
     ```bash
     nvcc -o rasterizer main.cu
     ```

2. **Run**:
   - Execute the program:
     ```bash
     ./rasterizer
     ```

3. **Output**:
   - The rendered image will be saved as a BMP file in the current directory.

---

## Key Concepts

- **Rasterization**:
  - Maps triangles in 3D space to 2D screen space.
  - Uses barycentric interpolation for smooth gradients and textures.

- **Barycentric Coordinates**:
  - Determines whether a pixel lies inside a triangle and computes interpolated attributes like color.

- **Texture Sampling**:
  - Maps texture coordinates to the closest texel and extracts normalized RGBA colors.

---

## Future Improvements

- Implement advanced lighting models for more realistic rendering.
- Add support for more complex textures and shading techniques.
- Optimize rasterization for larger screen resolutions.

---

## Acknowledgments

- NVIDIA CUDA Toolkit for GPU programming.
- Inspired by standard rasterization techniques in computer graphics.

---

## License

This project is open-source and available under the MIT License.
