
__kernel void simple(int N, __global float *x){

  int id = get_global_id(0);
  
  if(id<N)
    x[id] = id;

}
