/******************************************************************************
 * A simple CUDA “triangle + texture + Lambert lighting” software rasterizer,
 * akin to your Metal example. 
 *
 * Steps:
 *  1) Define 3 vertices (position, color, normal, texCoord).
 *  2) Define a small texture (or load from file). 
 *  3) For each pixel, do barycentric interpolation, Lambert lighting, 
 *     texture sampling, and output the final color to an RGBA buffer.
 *  4) Write out an 800x600 BMP (named "output.bmp").
 *
 * Compile & Run:
 *   nvcc main.cu -o triangle
 *   ./triangle
 * Check the generated output.bmp
 ******************************************************************************/

#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include "my_helper_math.h"

#define WIDTH  800
#define HEIGHT 600

// -----------------------------------------------------------------------------
// Data structures
// -----------------------------------------------------------------------------

// A structure matching your Metal vertex attributes: position, color, normal, texCoord
struct Vertex {
    float4 position;  // x,y,z,w
    float4 color;     // r,g,b,a
    float3 normal;    // nx,ny,nz
    float2 texCoord;  // u,v
};

// We’ll store the triangle in constant memory for easy device access.
__device__ __constant__ Vertex d_vertices[3];

// A small structure to hold texture data on the device
// We’ll do a simple RGBA 8-bit texture for demonstration.
struct TextureGpu {
    unsigned char* data;
    int width;
    int height;
};

__device__ __constant__ TextureGpu d_textureInfo;

// We’ll pass a light direction in constant memory, similar to your buffer(0).
__device__ __constant__ float3 d_lightDirection;

// -----------------------------------------------------------------------------
// Host-side data
// -----------------------------------------------------------------------------

// Three vertices
static Vertex h_vertices[3] = {
    // position          color                normal           texCoord
    // Top
    {{0.0f,  1.0f, 0.0f, 1.0f}, {1.0f,0.0f,0.0f,1.0f}, {0.0f, 0.5f, 1.0f}, {0.5f,1.0f}},
    // Bottom-left
    {{-1.0f, -1.0f, 0.0f,1.0f}, {0.0f,1.0f,0.0f,1.0f}, {-0.5f,-0.5f,1.0f},{0.0f,0.0f}},
    // Bottom-right
    {{1.0f,  -1.0f, 0.0f,1.0f}, {0.0f,0.0f,1.0f,1.0f}, {0.5f, -0.5f,1.0f}, {1.0f,0.0f}}
};

// Light direction (just a normal 3D vector)
static float3 h_lightDirection = {0.0f, 0.0f, 1.0f}; // from +Z toward the triangle

// -----------------------------------------------------------------------------
// Utility: Barycentric interpolation
// -----------------------------------------------------------------------------

__device__ inline void barycentric(
    float px, float py,
    float ax, float ay,
    float bx, float by,
    float cx, float cy,
    float &wA, float &wB, float &wC
) {
    float denom = (by - cy)*(ax - cx) + (cx - bx)*(ay - cy);
    wA = ((by - cy)*(px - cx) + (cx - bx)*(py - cy)) / denom;
    wB = ((cy - ay)*(px - cx) + (ax - cx)*(py - cy)) / denom;
    wC = 1.0f - wA - wB;
}

// -----------------------------------------------------------------------------
// Utility: Nearest-neighbor texture sampling
// -----------------------------------------------------------------------------

__device__ float4 sampleTexture(TextureGpu tex, float u, float v) 
{
    // Clamp or wrap as you see fit. Let’s clamp here.
    if(u < 0.0f) u = 0.0f; 
    if(v < 0.0f) v = 0.0f;
    if(u > 1.0f) u = 1.0f; 
    if(v > 1.0f) v = 1.0f;

    int x = static_cast<int>(u * (tex.width  - 1));
    int y = static_cast<int>(v * (tex.height - 1));

    // Each pixel in the texture is 4 bytes: [R, G, B, A]
    int idx = 4 * (y * tex.width + x);

    unsigned char r = tex.data[idx + 0];
    unsigned char g = tex.data[idx + 1];
    unsigned char b = tex.data[idx + 2];
    unsigned char a = tex.data[idx + 3];

    // Convert to float4 in [0..1]
    return make_float4(r/255.0f, g/255.0f, b/255.0f, a/255.0f);
}

// -----------------------------------------------------------------------------
// CUDA Kernel: Rasterize triangle with Lambert lighting & texturing
// -----------------------------------------------------------------------------

__global__ void rasterizeTriangle(unsigned char* outBuffer) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    // Access the triangle from constant memory
    Vertex v0 = d_vertices[0];
    Vertex v1 = d_vertices[1];
    Vertex v2 = d_vertices[2];

    // Light direction from constant memory
    float3 lightDir = d_lightDirection;

    // Clip->screen transform
    auto toScreenX = [](float clipX) {
        return 0.5f * (clipX + 1.0f) * (WIDTH - 1);
    };
    auto toScreenY = [](float clipY) {
        return 0.5f * (1.0f - clipY) * (HEIGHT - 1);
    };

    float ax = toScreenX(v0.position.x);
    float ay = toScreenY(v0.position.y);
    float bx = toScreenX(v1.position.x);
    float by = toScreenY(v1.position.y);
    float cx = toScreenX(v2.position.x);
    float cy = toScreenY(v2.position.y);

    // Pixel center
    float px = x + 0.5f;
    float py = y + 0.5f;

    // Barycentric
    float wA, wB, wC;
    barycentric(px, py, ax, ay, bx, by, cx, cy, wA, wB, wC);

    if (wA >= 0.0f && wB >= 0.0f && wC >= 0.0f) {

        // Interpolate the normal
        float3 normal = wA * v0.normal 
                     + wB * v1.normal 
                     + wC * v2.normal;

        // Normalize the normal
        float3 n = normalize(normal);

        // 2) Compute Lambert diffuse factor
        float diff = dot(n, lightDir);
        if (diff < 0.0f) diff = 0.0f;


        // 3) Interpolate the texture coordinate, sample the texture
        float2 uv = wA * v0.texCoord +
                    wB * v1.texCoord +
                    wC * v2.texCoord;
        float4 texColor = sampleTexture(d_textureInfo, uv.x, uv.y);

        // 4) Interpolate the vertex color
        float4 vertColor = wA * v0.color +
                           wB * v1.color +
                           wC * v2.color;

        // 5) Multiply: (Lambert) * (vertex color) * (texture color)
        float4 finalColor = diff * vertColor * texColor;

        // Force alpha to 1.0 or combine if you wish
        finalColor.w = 1.0f;

        // Clamp to [0,1]
        finalColor.x = fminf(fmaxf(finalColor.x, 0.0f), 1.0f);
        finalColor.y = fminf(fmaxf(finalColor.y, 0.0f), 1.0f);
        finalColor.z = fminf(fmaxf(finalColor.z, 0.0f), 1.0f);
        finalColor.w = fminf(fmaxf(finalColor.w, 0.0f), 1.0f);

        // Write final color
        int idx = 4 * (y * WIDTH + x);
        outBuffer[idx + 0] = static_cast<unsigned char>(finalColor.x * 255.0f);
        outBuffer[idx + 1] = static_cast<unsigned char>(finalColor.y * 255.0f);
        outBuffer[idx + 2] = static_cast<unsigned char>(finalColor.z * 255.0f);
        outBuffer[idx + 3] = static_cast<unsigned char>(finalColor.w * 255.0f);

    } 
    else {
        // Outside -> black
        int idx = 4 * (y * WIDTH + x);
        outBuffer[idx + 0] = 0;
        outBuffer[idx + 1] = 0;
        outBuffer[idx + 2] = 0;
        outBuffer[idx + 3] = 255; 
    }
}

// -----------------------------------------------------------------------------
// Write BMP
// -----------------------------------------------------------------------------

void writeBMP(const std::string &filename, const std::vector<unsigned char> &image, int width, int height)
{
    unsigned int fileSize = 54 + width * height * 4; 
    unsigned int dataOffset = 54;                    
    unsigned int headerSize = 40;                    
    unsigned short planes = 1;                       
    unsigned short bitsPerPixel = 32;                
    unsigned int compression = 0;                    
    unsigned int imageSize = width * height * 4;     
    unsigned int xPixelsPerM = 0, yPixelsPerM = 0;   
    unsigned int totalColors = 0;                    
    unsigned int importantColors = 0;               

    std::ofstream out(filename, std::ios::binary);
    // BMP File Header (14 bytes)
    out.put('B');
    out.put('M');
    out.write(reinterpret_cast<char*>(&fileSize), 4);
    unsigned int reserved = 0;
    out.write(reinterpret_cast<char*>(&reserved), 4);
    out.write(reinterpret_cast<char*>(&dataOffset), 4);

    // DIB Header (40 bytes)
    out.write(reinterpret_cast<char*>(&headerSize), 4);
    out.write(reinterpret_cast<char*>(&width), 4);
    out.write(reinterpret_cast<char*>(&height), 4);
    out.write(reinterpret_cast<char*>(&planes), 2);
    out.write(reinterpret_cast<char*>(&bitsPerPixel), 2);
    out.write(reinterpret_cast<char*>(&compression), 4);
    out.write(reinterpret_cast<char*>(&imageSize), 4);
    out.write(reinterpret_cast<char*>(&xPixelsPerM), 4);
    out.write(reinterpret_cast<char*>(&yPixelsPerM), 4);
    out.write(reinterpret_cast<char*>(&totalColors), 4);
    out.write(reinterpret_cast<char*>(&importantColors), 4);

    // Write pixels (BGRA for 32-bit)
    // Our buffer is RGBA top->bottom. BMP expects bottom->top in BGRA.
    for(int row = 0; row < height; row++) {
        int srcRow = height - 1 - row;
        int rowStart = srcRow * width * 4;
        for(int col = 0; col < width; col++) {
            int idx = rowStart + col*4;
            unsigned char r = image[idx + 0];
            unsigned char g = image[idx + 1];
            unsigned char b = image[idx + 2];
            unsigned char a = image[idx + 3];
            // BMP 32-bit expects BGRA
            out.put(b);
            out.put(g);
            out.put(r);
            out.put(a);
        }
    }
    out.close();
    std::cout << "Wrote " << filename << "\n";
}

// -----------------------------------------------------------------------------
// Example “loadTexture” that just generates a small checkerboard pattern
// In real usage, you might load a PNG with e.g. stb_image.
// -----------------------------------------------------------------------------
static const int TEXW = 256;
static const int TEXH = 256;
std::vector<unsigned char> createCheckerboardTexture()
{
    std::vector<unsigned char> tex(4*TEXW*TEXH);
    for(int y=0; y<TEXH; y++){
        for(int x=0; x<TEXW; x++){
            int idx = 4*(y*TEXW+x);
            bool c = ((x/32 + y/32) % 2) == 0;
            // White or black squares
            unsigned char val = c ? 255 : 0;
            tex[idx+0] = val;   // R
            tex[idx+1] = val;   // G
            tex[idx+2] = val;   // B
            tex[idx+3] = 255;   // A
        }
    }
    return tex;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main()
{
    // 1) Copy vertex data to constant memory
    cudaMemcpyToSymbol(d_vertices, h_vertices, 3*sizeof(Vertex));

    // 2) Copy light direction
    cudaMemcpyToSymbol(d_lightDirection, &h_lightDirection, sizeof(float3));

    // 3) Create a texture on CPU (checkerboard 256x256), or load from file
    std::vector<unsigned char> cpuTexture = createCheckerboardTexture();
    // In real usage, load a PNG with stb_image or similar.

    // 4) Allocate texture on GPU
    unsigned char *d_texData = nullptr;
    size_t texSize = cpuTexture.size() * sizeof(unsigned char);
    cudaMalloc(&d_texData, texSize);
    cudaMemcpy(d_texData, cpuTexture.data(), texSize, cudaMemcpyHostToDevice);

    // 5) Fill device texture info in constant memory
    TextureGpu texInfo;
    texInfo.data = d_texData;
    texInfo.width = TEXW;
    texInfo.height = TEXH;
    cudaMemcpyToSymbol(d_textureInfo, &texInfo, sizeof(TextureGpu));

    // 6) Allocate output color buffer on GPU
    size_t outSize = WIDTH * HEIGHT * 4 * sizeof(unsigned char);
    unsigned char* d_out = nullptr;
    cudaMalloc(&d_out, outSize);
    cudaMemset(d_out, 0, outSize);

    // 7) Launch kernel
    dim3 block(16, 16);
    dim3 grid((WIDTH+block.x-1)/block.x, (HEIGHT+block.y-1)/block.y);
    rasterizeTriangle<<<grid, block>>>(d_out);
    cudaDeviceSynchronize();

    // 8) Copy result back
    std::vector<unsigned char> h_out(outSize);
    cudaMemcpy(h_out.data(), d_out, outSize, cudaMemcpyDeviceToHost);

    // 9) Write BMP
    writeBMP("output.bmp", h_out, WIDTH, HEIGHT);

    // Cleanup
    cudaFree(d_out);
    cudaFree(d_texData);

    std::cout << "Done.\n";
    return 0;
}

