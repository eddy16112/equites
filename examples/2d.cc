#include <equites.h>
using namespace equites;

task(void, copyRegion, r_region<float, 2> r1, w_region<float, 2> r2){
  for(auto i : r1) r2.write(i, r1.read(i)); 
}

/* To make take advantage of polymorphism, we can define polymorphic functions
   and wrap monomorphic tasks around them. Plenty of room for improvement. */
template <class a, size_t ndim> 
void fill(w_region<a,ndim> r, a x){
  for(auto i : r) r.write(i, x);
}

task(void, fillRegion, w_region<float, 2> r, float x){
  fill(r, x); 
}

task(void, printRegion, r_region<float, 2> r){
  for(auto i : r){
    cout << "point " << i << " has value " << r.read(i) << endl; 
  }
}

task(void, toplevel){
  auto r1 = region(float, 2, point(2, 2));
  auto r2 = region(float, 2, point(2, 2));
  call(fillRegion, r1, 3.14159); 
  call(copyRegion, r1, r2);
  call(printRegion, r1); 
}

int main(int argc, char** argv){
  start(toplevel, argc, argv);
}
