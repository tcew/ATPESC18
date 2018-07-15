
__kernel void unrolledPartialReduce(const int entries,
				    __global const datafloat *u,
				    __global const datafloat *newu,
				    __global datafloat *red){

  __local datafloat s_red[BDIM];

  const int id = get_global_id(0);
  const int tid = get_local_id(0);

  s_red[tid] = 0;

  if(id < entries){
    const datafloat diff = u[id] - newu[id];
    s_red[tid] = diff*diff;
  }

  barrier(CLK_LOCAL_MEM_FENCE);  // barrier (make sure s_red is ready)

  // manually unrolled reduction (assumes BDIM=256)
  if(BDIM>128) {
    if(tid<128)
      s_red[tid] += s_red[tid+128];

    barrier(CLK_LOCAL_MEM_FENCE);  // barrier (make sure s_red is ready)
  }

  if(BDIM>64){
    if(tid<64)
      s_red[tid] += s_red[tid+64];

    barrier(CLK_LOCAL_MEM_FENCE);  // barrier (make sure s_red is ready)
  }

  if(BDIM>32){
    if(tid<32)
      s_red[tid] += s_red[tid+32];

    barrier(CLK_LOCAL_MEM_FENCE);  // barrier (make sure s_red is ready)
  }

  if(BDIM>16){
    if(tid<16)
      s_red[tid] += s_red[tid+16];

    barrier(CLK_LOCAL_MEM_FENCE);  // barrier (make sure s_red is ready)
  }

  if(BDIM>8){
    if(tid<8)
      s_red[tid] += s_red[tid+8];

    barrier(CLK_LOCAL_MEM_FENCE);  // barrier (make sure s_red is ready)
  }

  if(BDIM>4){
    if(tid<4)
      s_red[tid] += s_red[tid+4];

    barrier(CLK_LOCAL_MEM_FENCE);  // barrier (make sure s_red is ready)
  }

  if(BDIM>2){
    if(tid<2)
      s_red[tid] += s_red[tid+2];

    barrier(CLK_LOCAL_MEM_FENCE);  // barrier (make sure s_red is ready)
  }

  if(BDIM>1){
    if(tid<1)
      s_red[tid] += s_red[tid+1];
  }

  // store result of this block reduction
  if(tid==0)
    red[get_group_id(0)] = s_red[tid];
}
