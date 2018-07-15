#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#ifdef __APPLE__
#include <OpenCL/OpenCl.h>
#else
#include <CL/cl.h>
#endif

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
  fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

int main(int argc, char **argv){

  int plat = 0;
  int dev  = 0;

  /* set up CL */
  cl_int            err;
  cl_platform_id    platforms[100];
  cl_uint           platforms_n;
  cl_device_id      devices[100];
  cl_uint           devices_n ;

  cl_context        context;
  cl_command_queue  queue;
  cl_device_id      device;

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
  
#if 1
  // build kernel function
  const char *sourceFileName = "simple.cl";
  const char *functionName = "simple";

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
#else
  
  const char *source =
    "__kernel void simple(int N, __global float *x){"
    "                                            "
    "     int id = get_global_id(0);             "
    "                                            "
    "     if(id<N)                               "
    "       x[id] = id;                          "
    "                                            "
    "}";

  const char *functionName = "foo";
#endif

  /* create program from source */
  cl_program program = clCreateProgramWithSource(context, 1, (const char **) & source, (size_t*) NULL, &err);

  if (!program){
    printf("Error: Failed to create compute program!\n");
    throw 1;
  }
    
  /* compile and build program */
  const char *allFlags = " ";
  err = clBuildProgram(program, 1, &device, allFlags, (void (*)(cl_program, void*))  NULL, NULL);

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
  cl_kernel kernel = clCreateKernel(program, functionName, &err);
  if (! kernel || err != CL_SUCCESS){
    printf("Error: Failed to create compute kernel!\n");
    throw 1;
  }
  
  int N = 100; /* vector size  */

  /* create host array */
  size_t sz = N*sizeof(float);

  float *h_x = (float*) malloc(sz);
  for(int n=0;n<N;++n)
    h_x[n] = n;

  /* create device buffer and copy from host buffer */
  cl_mem c_x = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, h_x, &err);

  /* now set kernel arguments */
  clSetKernelArg(kernel, 0, sizeof(int), &N);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &c_x);
  
  /* set thread array */
  int dim = 1;
  int Nt = 10;
  int Ng = Nt*((N+Nt-1)/Nt);
  size_t local[3] = {Nt,1,1};
  size_t global[3] = {Ng,1,1};

  /* queue up kernel */
  clEnqueueNDRangeKernel(queue, kernel, dim, 0, global, local, 0, (cl_event*)NULL, NULL);

  /* blocking read from device to host */
  clFinish(queue);
  
  /* blocking read to host */
  clEnqueueReadBuffer(queue, c_x, CL_TRUE, 0, sz, h_x, 0, 0, 0);
  
  /* print out results */
  for(int n=0;n<N;++n)
    printf("h_x[%d] = %g\n", n, h_x[n]);

  exit(0);
  
}
