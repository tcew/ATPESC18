#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#ifdef __APPLE__
#include <OpenCL/OpenCl.h>
#else
#include <CL/cl.h>
#endif

#define datafloat float

#define BX 16
#define BY 16
#define BDIM 256

// user supplied error call back function 
void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
  fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}


char *readFileToString(const char *sourceFileName){

  // read in text from source file
  struct stat statbuf;
  FILE *fh = fopen(sourceFileName, "r");
  if (fh == 0){
    printf("Failed to open: %s\n", sourceFileName);
    throw 1;
  }
  /* get stats for source file */
  stat(sourceFileName, &statbuf);

  /* read text from source file and add terminator */
  char *source = (char *) malloc(statbuf.st_size + 1);
  fread(source, statbuf.st_size, 1, fh);
  source[statbuf.st_size] = '\0';

  return source;
}

cl_kernel buildKernelFromSource(cl_context &context, cl_device_id &device, 
				const char *sourceFileName, const char *functionName, const char *compilerFlags){

  cl_int  err;

  char *source = readFileToString(sourceFileName);

  /* create program from source */
  cl_program program = clCreateProgramWithSource(context, 1, (const char **) & source, (size_t*) NULL, &err);

  if (!program){
    printf("Error: Failed to create compute program!\n");
    throw 1;
  }
    
  /* compile and build program */
  err = clBuildProgram(program, 1, &device, compilerFlags, (void (*)(cl_program, void*))  NULL, NULL);

  /* check for compilation errors */
  char *build_log;
  size_t ret_val_size;
  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
  
  build_log = (char*) malloc(ret_val_size+1);
  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, (size_t*) NULL);
  
  /* to be carefully, terminate with \0
     there's no information in the reference whether the string is 0 terminated or not */
  build_log[ret_val_size] = '\0';

  /* print out compilation log */
  fprintf(stderr, "%s", build_log );

  /* create runnable kernel */
  cl_kernel *kernel = (cl_kernel*) calloc(1,sizeof(cl_kernel));
  *kernel = clCreateKernel(program, functionName, &err);
  if (! *kernel || err != CL_SUCCESS){
    printf("Error: Failed to create compute kernel!\n");
    throw 1;
  }

  return kernel[0];

}


void solve(int N, datafloat tol, datafloat *h_rhs, datafloat *h_res, datafloat *h_u, datafloat *h_u2 ){

  const int iterationChunk = 100; // needs to be multiple of 2
  int iterationsTaken = 0;

  /* set up CL */
  cl_int            err;
  cl_platform_id    platforms[100];
  cl_uint           platforms_n;
  cl_device_id      devices[100];
  cl_uint           devices_n ;

  cl_context        context;
  cl_command_queue  queue;
  cl_device_id      device;

  /* hard code platform and device number */
  int plat = 0;
  int dev = 1;

  /* get list of platform IDs (platform == implementation of OpenCL) */
  clGetPlatformIDs(100, platforms, &platforms_n);
  
  if( plat > platforms_n) {
    printf("ERROR: platform %d unavailable \n", plat);
    exit(-1);
  }
  
  // find all available device IDs on chosen platform (could restrict to CPU or GPU)
  cl_uint dtype = CL_DEVICE_TYPE_ALL;
  clGetDeviceIDs( platforms[plat], dtype, 100, devices, &devices_n);
  
  printf("devices_n = %d\n", devices_n);
  
  if(dev>=devices_n){
    printf("invalid device number for this platform\n");
    exit(0);
  }

  // choose user specified device
  device = devices[dev];
  
  // make compute context on device
  context = clCreateContext((cl_context_properties *)NULL, 1, &device, &pfn_notify, (void*)NULL, &err);

  // create command queue
  queue   = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  
  // build jacobi kernel from source file
  const char *functionName = "";

  char flags[BUFSIZ];
  if(sizeof(datafloat)==sizeof(float))
    sprintf(flags, "-DBX=%d -DBY=%d -DBDIM=%d -Ddatafloat=float", BX, BY, BDIM);
  if(sizeof(datafloat)==sizeof(double))
    sprintf(flags, "-DBX=%d -DBY=%d -DBDIM=%d -Ddatafloat=double", BX, BY, BDIM);

  // build Jacobi kernel
  cl_kernel jacobi        = buildKernelFromSource(context, device, "jacobi.cl", "jacobi", flags);

  // build partial reduction kernel
  cl_kernel partialReduce = buildKernelFromSource(context, device, "partialReduce.cl", "partialReduce", flags);

  // build Device Arrays and transfer data from host arrays
  size_t sz = (N+2)*(N+2)*sizeof(datafloat);
  cl_mem c_u   = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz,   h_u, &err);
  cl_mem c_u2  = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz,  h_u2, &err);
  cl_mem c_rhs = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, h_rhs, &err);
  cl_mem c_res = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, h_res, &err);

  datafloat res;

  do {

    // Call jacobi [iterationChunk] times before calculating residual
    for(int i = 0; i < iterationChunk/2; ++i){

      // design thread array for Jacobi iteration
      int dim = 2;
      size_t local[3] = {BX,BY,1};
      size_t global[3] = {BX*((N+BX-1)/BX),BY*((N+BY-1)),1};

      clSetKernelArg(jacobi, 0, sizeof(int), &N);
      clSetKernelArg(jacobi, 1, sizeof(cl_mem), &c_rhs);

      // first iteration  - note order of arguments
      clSetKernelArg(jacobi, 2, sizeof(cl_mem), &c_u);
      clSetKernelArg(jacobi, 3, sizeof(cl_mem), &c_u2);

      clEnqueueNDRangeKernel(queue, jacobi, dim, 0, global, local, 0, (cl_event*)NULL, NULL);

      // second iteration - switch order of arguments
      clSetKernelArg(jacobi, 2, sizeof(cl_mem), &c_u2);
      clSetKernelArg(jacobi, 3, sizeof(cl_mem), &c_u);

      clEnqueueNDRangeKernel(queue, jacobi, dim, 0, global, local, 0, (cl_event*)NULL, NULL);

    }
    
    // calculate norm(u-u2)
    {
      // design thread array for norm(u-u2)
      int N2 = (N+2)*(N+2);
      int dim = 1;
      int Nred = (N2+BDIM-1)/BDIM;
      size_t local[3] = {BDIM,1,1};
      size_t global[3] = {BDIM*Nred,1,1};

      clSetKernelArg(partialReduce, 0, sizeof(int), &N2);
      clSetKernelArg(partialReduce, 1, sizeof(cl_mem), &c_u);
      clSetKernelArg(partialReduce, 2, sizeof(cl_mem), &c_u2);
      clSetKernelArg(partialReduce, 3, sizeof(cl_mem), &c_res);
      
      clEnqueueNDRangeKernel(queue, partialReduce, dim, 0, global, local, 0, (cl_event*)NULL, NULL);
      
      // blocking copy from device to host 
      size_t sz = sizeof(datafloat)*Nred;
      clEnqueueReadBuffer(queue, c_res, CL_TRUE, 0, sz, h_res, 0, 0, 0);
      
      res = 0;
      for(int i = 0; i < Nred; ++i)
	res += h_res[i];

      res = sqrt(res);
    }
    
    iterationsTaken += iterationChunk;
    
    printf("residual = %g after %d steps \n", res, iterationsTaken);

  } while(res > tol);

  printf("Residual                   : %7.9e\n"     , res);

  {
    // blocking copy of solution from device to host 
    size_t sz = sizeof(datafloat)*(N+2)*(N+2);
    clEnqueueReadBuffer(queue, c_u, CL_TRUE, 0, sz, h_u, 0, 0, 0);
  }

  // free device arrays
  clReleaseMemObject(c_u);
  clReleaseMemObject(c_u2);
  clReleaseMemObject(c_rhs);
  clReleaseMemObject(c_res);

}

int main(int argc, char** argv){

  // parse command line arguements
  if(argc != 3){
    printf("Usage: ./main N tol \n");
    return 0;
  }

  // Number of internal domain nodes in each direction
  const int N     = atoi(argv[1]);

  // Termination criterion || unew - u ||_2 < tol 
  const datafloat tol = atof(argv[2]);

  // Host Arrays
  datafloat *h_u   = (datafloat*) calloc((N+2)*(N+2), sizeof(datafloat));
  datafloat *h_u2  = (datafloat*) calloc((N+2)*(N+2), sizeof(datafloat));
  datafloat *h_rhs = (datafloat*) calloc((N+2)*(N+2), sizeof(datafloat));
  datafloat *h_res = (datafloat*) calloc((N+2)*(N+2), sizeof(datafloat));

  // FD node spacing
  datafloat delta = 2./(N+1);

  for(int j = 0; j < N+2; ++j){
    for(int i = 0; i < N+2; ++i){
      datafloat x = -1 + delta*i;
      datafloat y = -1 + delta*j;

      // solution is u = sin(pi*x)*sin(pi*y) so the rhs is: 
      h_rhs[i + (N+2)*j] = delta*delta*(2.*M_PI*M_PI*sin(M_PI*x)*sin(M_PI*y));
    }
  }

  // Solve discrete Laplacian
  solve(N, tol, h_rhs, h_res, h_u, h_u2);

  // Compute maximum error in finite difference solution and output solution
  FILE *fp = fopen("result.dat", "w");
  datafloat maxError = 0;
  for(int j = 0; j < N+2; ++j){
    for(int i = 0; i < N+2; ++i){
      datafloat x = -1 + delta*i;
      datafloat y = -1 + delta*j;
      datafloat error = fabs( sin(M_PI*x)*sin(M_PI*y) - h_u[i + (N+2)*j]);
      maxError = (error > maxError) ? error:maxError;
      fprintf(fp, "%g %g %g %g\n", x, y, h_u[i+(N+2)*j],error);
    }
  }
  fclose(fp);

  printf("Maximum absolute error     : %7.9e\n"     ,  maxError);

  // Free all the mess
  free(h_u);
  free(h_u2);
  free(h_res);
  free(h_rhs);
}
