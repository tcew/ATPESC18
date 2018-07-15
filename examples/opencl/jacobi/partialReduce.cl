
__kernel void partialReduce(const int entries,
			    __global const datafloat *u,
			    __global const datafloat *newu,
			    __global datafloat *blocksum){

  // local memory for scratch pad
  __local datafloat s_blocksum[BDIM];

  // global thread index
  const int id = get_global_id(0);

  // number of threads in thread block (all alive initially)
  int alive = get_local_size(0);

  // local thread index
  int t = get_local_id(0);

  // load block of vector into shared memory
  s_blocksum[t] = 0;

  // load global data into local memory if in range
  if(id < entries){
    const datafloat diff = u[id] - newu[id];
    s_blocksum[t] = diff*diff;
  }

  // iterate while there is more than one thread alive
  while(alive>1){

    barrier(CLK_LOCAL_MEM_FENCE);  // barrier (make sure s_blocksum is ready) 
    
    // kill half of the threads in the work-group
    alive /= 2;

    // if this thread is alive then do sum
    if(t < alive) s_blocksum[t] += s_blocksum[t+alive];
  }

  // last thread standing does write out
  if(t==0)
    blocksum[get_group_id(0)] = s_blocksum[0];
}

