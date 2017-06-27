#include <equites.h>
using namespace equites;

task(void, triad, size_t iters, float x, r_region<float, 1> a, r_region<float, 1> b, w_region<float, 1> c){
  for(size_t iter = 0; iter < iters; iter++){
    for(auto i : a){
      c.write(i, a.read(i) * x + b.read(i));
    }
  }
}

task(void, fillRegion, w_region<float, 1> a, float x){
  for(auto i : a) a.write(i, x);
}

task(void, toplevel, int argc, char** argv){
  size_t size = argc > 1 ? atol(argv[1]) : 1000000;
  size_t ntask = argc > 2 ? atol(argv[2]) : 4; 
  size_t iters = argc > 3 ? atol(argv[3]) : 10; 
  for(size_t i=0; i<ntask; i++){
    auto a = region(float, 1, size); 
    auto b = region(float, 1, size); 
    auto c = region(float, 1, size); 
    call(fillRegion, a, 3.14159);
    call(fillRegion, b, 1.71828);
    call(triad, iters, 1.61803, a, b, c);
  }
  cout << "Running " << size * ntask * iters * 3 * sizeof(float) << " memory operations" << endl; 
}

int main(int argc, char** argv){
  start(toplevel, argc, argv); 
}

  
