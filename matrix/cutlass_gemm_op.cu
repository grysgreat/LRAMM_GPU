#include "cutlass_gemm_op.cuh"


void cut_gemm(input_t *A, input_t* B, output_t* C,int rowA,int colA, int rowB,int colB){

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(rowA, colB, rowB);


  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);  


  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     {A,colA},  // <- reference to matrix A on device
                                     {B,colA},  // <- reference to matrix B on device
                                     {C,colB},  // <- reference to matrix C on device
                                     {C,colB},  // <- reference to matrix D on device
                                     {alpha, beta}       // <- tuple of alpha and beta
                                     };        // <- k-dimension split factor


  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<input_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
  gemm_op();

}
void cut_gemm4(cutlass::int4b_t *A, cutlass::int4b_t* B, int32_t* C,int rowA,int colA, int rowB,int colB){

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(rowA, colB, rowB);


  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);  

  int split_k_slices = 1;
  typename Gemm4::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     {A,colA},  // <- reference to matrix A on device
                                     {B,colA},  // <- reference to matrix B on device
                                     {C,colB},  // <- reference to matrix C on device
                                     {C,colB},  // <- reference to matrix D on device
                                     {alpha, beta},       // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor


  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm4::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<input_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm4 gemm_op;

  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
  gemm_op();

}

void cut_gemm_workspace(input_t *A, input_t* B, output_t* C,input_t* workspace,int rowA,int colA, int rowB,int colB){

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(rowA, colB, rowB);


  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);  


  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     {A,colA},  // <- reference to matrix A on device
                                     {B,colA},  // <- reference to matrix B on device
                                     {C,colB},  // <- reference to matrix C on device
                                     {C,colB},  // <- reference to matrix D on device
                                     {alpha, beta}       // <- tuple of alpha and beta
                                     };        // <- k-dimension split factor


  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;
  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status = gemm_op.initialize(arguments, workspace);
  gemm_op();

}



void cut_gemm_workspace4(cutlass::int4b_t *A, cutlass::int4b_t* B, int32_t* C,cutlass::int4b_t* workspace,int rowA,int colA, int rowB,int colB){

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(rowA, colB, rowB);


  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);  


  typename Gemm4::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     {A,colA},  // <- reference to matrix A on device
                                     {B,colA},  // <- reference to matrix B on device
                                     {C,colB},  // <- reference to matrix C on device
                                     {C,colB},  // <- reference to matrix D on device
                                     {alpha, beta}       // <- tuple of alpha and beta
                                     };        // <- k-dimension split factor


  // Instantiate CUTLASS kernel depending on templates
  Gemm4 gemm_op;
  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status = gemm_op.initialize(arguments, workspace);
  gemm_op();

}
#define BLOCK_SIZE 32

__global__ void cuda_transpose(cutlass::int4b_t *matrix,cutlass::int4b_t *tr_matrix,int m,int n) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ cutlass::int4b_t smem_matrix[BLOCK_SIZE][BLOCK_SIZE];
	smem_matrix[threadIdx.y][threadIdx.x] = row < m&& col < n ?  static_cast<cutlass::int4b_t>(matrix[row*n+col]) :  static_cast<cutlass::int4b_t>(0);
	__syncthreads();
	if(blockIdx.x * blockDim.x + threadIdx.y < n && threadIdx.x + blockIdx.y * blockDim.x < m)
	tr_matrix[threadIdx.x+blockIdx.y*blockDim.x+m*(blockIdx.x*blockDim.x+threadIdx.y)] = smem_matrix[threadIdx.x][threadIdx.y];
	return;
}
void I4trans(cutlass::int4b_t *odata, cutlass::int4b_t *idata,int rows,int cols){
    // dim3 dimGrid(rows/TILE_DIM, cols/TILE_DIM, 1);
    // dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);    

    // I8trans_cuda<<<dimGrid,dimBlock>>>(odata,idata);
    // cudaDeviceSynchronize();

	dim3 gird = { (unsigned int)(cols - 1 + BLOCK_SIZE) / BLOCK_SIZE, (unsigned int)(rows - 1 + BLOCK_SIZE) / BLOCK_SIZE,1 };
	dim3 block = { BLOCK_SIZE,BLOCK_SIZE,1 };

	cuda_transpose << < gird , block  >> > (idata, odata, rows, cols);
    cudaDeviceSynchronize();
}